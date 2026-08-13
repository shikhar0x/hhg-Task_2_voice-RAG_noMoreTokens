import os
from config.settings import settings
from config.logger import logger
from harness.base import BaseStep, StepResult
from harness.retry import retry_step
from generation.prompts import SYSTEM_RAG_PROMPT, format_user_prompt

class LLMGenerationStep(BaseStep):
    """Ultra-fast grounded LLM generation with Groq / Ollama / Extractive fallback."""
    name = "generation_llm"

    def __init__(self):
        self.groq_client = None
        if settings.groq_api_key:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=settings.groq_api_key)
            except Exception as e:
                logger.warning(f"Could not initialize Groq client: {e}")

    @retry_step(max_retries=2, base_delay=0.05, max_delay=0.2)
    def _generate_groq(self, prompt: str) -> str:
        resp = self.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_RAG_PROMPT},
                {"role": "user", "content": prompt}
            ],
            model=settings.groq_model,
            temperature=0.1,
            max_tokens=90
        )
        return resp.choices[0].message.content.strip()

    def execute(self, input_data: dict) -> StepResult:
        question = input_data.get("transcript", "")
        context = input_data.get("context", "")

        if not question:
            return StepResult(success=False, error="No query supplied to LLM step.")

        full_prompt = format_user_prompt(question, context)

        # 1. Use Groq if API key is provided (Ultra-fast ~60-100ms)
        if self.groq_client:
            try:
                answer = self._generate_groq(full_prompt)
                return StepResult(success=True, data={"answer": answer, "provider": "groq"})
            except Exception as e:
                logger.warning(f"Groq generation failed: {e}. Falling back...")

        # 2. Fast extractive fallback if offline / testing
        lines = [line for line in context.split("\n") if line and not line.startswith("Passage")]
        concise_ans = " ".join(lines)[:220] + "." if lines else context[:220]
        return StepResult(
            success=True,
            data={"answer": f"Grounded Answer: {concise_ans}", "provider": "extractive_fast"}
        )
