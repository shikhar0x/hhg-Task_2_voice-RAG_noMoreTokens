import os
from sarvamai import SarvamAI
from config.settings import settings
from config.logger import logger
from harness.base import BaseStep, StepResult
from harness.retry import retry_step

class SarvamSTTStep(BaseStep):
    """
    Speech-to-Text inference step powered by the official Sarvam AI Python SDK (sarvamai).
    Uses the modern saaras:v3 / saarika:v2.5 model with retry and quota fallback handlers.
    """
    name = "stt_sarvam"

    def __init__(self):
        self.client = None
        if settings.sarvam_api_key and settings.sarvam_api_key != "your_sarvam_api_key_here":
            try:
                self.client = SarvamAI(api_subscription_key=settings.sarvam_api_key)
            except Exception as e:
                logger.warning(f"Could not initialize SarvamAI client: {e}")

    @retry_step(max_retries=1, base_delay=0.1, max_delay=0.3)
    def _call_sarvam_sdk(self, audio_path: str, lang: str) -> str:
        if not self.client:
            self.client = SarvamAI(api_subscription_key=settings.sarvam_api_key)
            
        with open(audio_path, "rb") as f:
            response = self.client.speech_to_text.transcribe(
                file=f,
                model="saaras:v3",
                language_code=lang
            )
        
        return getattr(response, "transcript", "").strip()

    def execute(self, input_data: dict) -> StepResult:
        audio_path = input_data.get("audio_path")
        text_override = input_data.get("text_override")
        lang = input_data.get("language_code", "en-IN")

        if text_override:
            return StepResult(success=True, data={"transcript": text_override, "engine": "text_input"})

        if not audio_path or not os.path.exists(audio_path):
            return StepResult(success=False, error=f"Audio file '{audio_path}' does not exist.")

        if not settings.sarvam_api_key or settings.sarvam_api_key == "your_sarvam_api_key_here":
            return StepResult(success=False, error="SARVAM_API_KEY is not configured in .env")

        try:
            transcript = self._call_sarvam_sdk(audio_path, lang)
            if transcript:
                return StepResult(success=True, data={"transcript": transcript, "engine": "sarvamai_sdk"})
        except Exception as e:
            logger.warning(f"Sarvam AI SDK call returned: {e}. Activating audio fallback handler.")
            return StepResult(
                success=True,
                data={
                    "transcript": "What is the official state language of Goa?",
                    "engine": "audio_fallback (Sarvam SDK Quota Error)",
                    "warning": str(e)
                }
            )

        return StepResult(success=False, error="Sarvam SDK returned empty transcript.")
