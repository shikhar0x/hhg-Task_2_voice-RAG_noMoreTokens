import ollama
from config.settings import settings
from config.logger import logger
from harness.base import BaseStep, StepResult
from harness.retry import retry_step
from generation.prompts import SYSTEM_RAG_PROMPT, format_user_prompt

class LLMGenerationStep(BaseStep):
    """Streaming & non-streaming grounded LLM generation (adapted from Nexa runtime)."""
    name = "generation_llm"

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.ollama_model

    @retry_step(max_retries=2, base_delay=0.05, max_delay=0.3)
    def _call_ollama(self, prompt: str) -> str:
        client = ollama.Client(host=settings.ollama_host)
        resp = client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_RAG_PROMPT},
                {"role": "user", "content": prompt}
            ],
            options={"temperature": 0.1, "num_predict": 128}
        )
        return resp.get("message", {}).get("content", "").strip()

    def execute(self, input_data: dict) -> StepResult:
        question = input_data.get("transcript", "")
        context = input_data.get("context", "")

        if not question:
            return StepResult(success=False, error="No query supplied to LLM step.")

        full_prompt = format_user_prompt(question, context)

        try:
            answer = self._call_ollama(full_prompt)
            return StepResult(success=True, data={"answer": answer, "model": self.model_name})
        except Exception as e:
            logger.warning(f"Ollama call failed or unavailable ({e}). Falling back to fast extractive summary.")
            # Fast extractive fallback ensures pipeline finishes reliably under demo constraints
            fallback_answer = f"Based on retrieved context: {context[:250]}..."
            return StepResult(
                success=True,
                data={"answer": fallback_answer, "model": "extractive_fallback", "warning": str(e)}
            )
