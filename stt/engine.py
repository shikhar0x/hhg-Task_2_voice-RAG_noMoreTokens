import os
import requests
from dotenv import load_dotenv
from config.settings import settings
from config.logger import logger
from harness.base import BaseStep, StepResult
from harness.retry import retry_step

load_dotenv(override=True)

class SpeechToTextStep(BaseStep):
    """
    Generalized Speech-to-Text pipeline step supporting official providers:
    - ElevenLabs (scribe_v2)
    - Sarvam AI (saaras:v3 / saarika:v2.5)
    """
    name = "stt_engine"

    def _call_elevenlabs(self, audio_path: str) -> str:
        api_key = os.getenv("ELEVENLABS_API_KEY") or settings.elevenlabs_api_key
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY not configured")

        url = "https://api.elevenlabs.io/v1/speech-to-text"
        headers = {"xi-api-key": api_key}
        filename = os.path.basename(audio_path)
        with open(audio_path, "rb") as f:
            files = {"file": (filename, f, "audio/wav")}
            data = {"model_id": "scribe_v2"}
            resp = requests.post(url, headers=headers, files=files, data=data, timeout=8.0)

        if resp.status_code == 200:
            return resp.json().get("text", "").strip()
        raise RuntimeError(f"ElevenLabs error {resp.status_code}: {resp.text}")

    def _call_sarvam(self, audio_path: str, lang: str) -> str:
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

        # 1. Try ElevenLabs
        try:
            transcript = self._call_elevenlabs(audio_path)
            if transcript:
                return StepResult(success=True, data={"transcript": transcript, "engine": "elevenlabs_scribe_v2"})
        except Exception as e:
            logger.warning(f"ElevenLabs STT attempt note: {e}")

        # 2. Try Sarvam AI
        try:
            transcript = self._call_sarvam(audio_path, lang)
            if transcript:
                return StepResult(success=True, data={"transcript": transcript, "engine": "sarvam_saaras_v3"})
        except Exception as e:
            logger.warning(f"Sarvam AI STT attempt note: {e}")

        return StepResult(
            success=False,
            error="STT failed: Please check ELEVENLABS_API_KEY or SARVAM_API_KEY in .env"
        )
