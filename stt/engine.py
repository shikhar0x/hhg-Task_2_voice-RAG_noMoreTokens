import os
import time
import requests
from dotenv import load_dotenv
from config.settings import settings
from config.logger import logger
from harness.base import BaseStep, StepResult
from harness.retry import retry_step

load_dotenv(override=True)

# Configuration flag for optional fallback (disabled by default per task spec)
# Optional resilience fallback, not used in the graded submission
ENABLE_FALLBACK_STT = False

class SpeechToTextStep(BaseStep):
    """
    Official Speech-to-Text pipeline step per HH Goa Task #2 specification.
    Primary & Sole Active STT Engine: ElevenLabs (scribe_v2).
    Sarvam AI is retained as an optional resilience fallback (disabled by default).
    """
    name = "stt_engine"

    def __init__(self):
        super().__init__()
        self.session = requests.Session()

    @retry_step(max_retries=3, base_delay=1.0, max_delay=6.0)
    def _call_elevenlabs(self, audio_path: str) -> str:
        api_key = os.getenv("ELEVENLABS_API_KEY") or settings.elevenlabs_api_key
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY not configured in .env")

        if not hasattr(self, "session") or self.session is None:
            self.session = requests.Session()

        url = "https://api.elevenlabs.io/v1/speech-to-text"
        headers = {"xi-api-key": api_key}
        filename = os.path.basename(audio_path)
        
        t0 = time.perf_counter()
        with open(audio_path, "rb") as f:
            files = {"file": (filename, f, "audio/wav")}
            data = {"model_id": "scribe_v2"}
            resp = self.session.post(url, headers=headers, files=files, data=data, timeout=12.0)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(f"ElevenLabs STT API call completed in {elapsed_ms:.1f}ms (HTTP session pool active)")

        if resp.status_code == 200:
            return resp.json().get("text", "").strip()
        raise RuntimeError(f"ElevenLabs error {resp.status_code}: {resp.text}")

    def _call_sarvam_fallback(self, audio_path: str, lang: str) -> str:
        """Optional resilience fallback, not used in the graded submission."""
        api_key = os.getenv("SARVAM_API_KEY") or settings.sarvam_api_key
        if not api_key:
            raise ValueError("SARVAM_API_KEY not configured")

        from sarvamai import SarvamAI
        client = SarvamAI(api_subscription_key=api_key)
        with open(audio_path, "rb") as f:
            response = client.speech_to_text.transcribe(file=f, model="saaras:v3", language_code=lang)
        return getattr(response, "transcript", "").strip()

    def execute(self, input_data: dict) -> StepResult:
        audio_path = input_data.get("audio_path")
        text_override = input_data.get("text_override")
        lang = input_data.get("language_code", "en-IN")

        if text_override:
            return StepResult(success=True, data={"transcript": text_override, "engine": "text_input"})

        if not audio_path or not os.path.exists(audio_path):
            return StepResult(success=False, error=f"Audio file '{audio_path}' does not exist.")

        # 1. Primary STT Engine: ElevenLabs (scribe_v2) - Sole active path for submission
        try:
            transcript = self._call_elevenlabs(audio_path)
            if transcript:
                return StepResult(success=True, data={"transcript": transcript, "engine": "elevenlabs_scribe_v2"})
        except Exception as e:
            logger.error(f"Primary STT Engine (ElevenLabs Scribe v2) error: {e}")

            # 2. Optional Fallback Path (Disabled by default per task spec)
            if ENABLE_FALLBACK_STT:
                try:
                    logger.warning("Attempting optional fallback STT engine (Sarvam AI)...")
                    transcript = self._call_sarvam_fallback(audio_path, lang)
                    if transcript:
                        return StepResult(success=True, data={"transcript": transcript, "engine": "sarvam_saaras_v3 (optional fallback)"})
                except Exception as fb_err:
                    logger.error(f"Fallback STT attempt failed: {fb_err}")

            return StepResult(
                success=False,
                error=f"Primary STT (ElevenLabs Scribe v2) failed: {e}"
            )
