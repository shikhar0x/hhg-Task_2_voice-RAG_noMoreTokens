import os
import requests
from config.settings import settings
from config.logger import logger
from harness.base import BaseStep, StepResult
from harness.retry import retry_step

class SarvamSTTStep(BaseStep):
    """Sarvam AI STT harness step (saarika:v2) with retry resilience & mock fallback."""
    name = "stt_sarvam"

    @retry_step(max_retries=3, base_delay=0.05, max_delay=0.4)
    def _call_sarvam_api(self, audio_path: str, lang: str) -> str:
        headers = {"api-subscription-key": settings.sarvam_api_key}
        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
            data = {"model": "saarika:v2", "language_code": lang}
            resp = requests.post(settings.sarvam_stt_url, headers=headers, files=files, data=data, timeout=4.0)

        if resp.status_code == 200:
            return resp.json().get("transcript", "")
        raise RuntimeError(f"Sarvam STT failed with status {resp.status_code}: {resp.text}")

    def execute(self, input_data: dict) -> StepResult:
        audio_path = input_data.get("audio_path")
        text_override = input_data.get("text_override")
        lang = input_data.get("language_code", "en-IN")

        # Allow instant mock / text injection for testing without live microphone
        if text_override:
            return StepResult(success=True, data={"transcript": text_override, "engine": "mock_override"})

        if not audio_path or not os.path.exists(audio_path):
            return StepResult(success=False, error=f"Audio file '{audio_path}' does not exist.")

        # Fallback simulation if API key is not yet set
        if settings.sarvam_api_key == "demo_sarvam_key":
            logger.warning("SARVAM_API_KEY is not configured. Simulating speech transcription for local testing.")
            return StepResult(
                success=True,
                data={"transcript": "What is the capital of Goa and its official language?", "engine": "simulation"}
            )

        transcript = self._call_sarvam_api(audio_path, lang)
        return StepResult(success=True, data={"transcript": transcript, "engine": "sarvam_saarika_v2"})
