import os
from dotenv import load_dotenv
from config.settings import settings
from config.logger import logger
from harness.base import BaseStep, StepResult
from harness.retry import retry_step
from generation.prompts import SYSTEM_RAG_PROMPT, format_user_prompt

# Ensure .env is reloaded with GROQ_API_KEY
load_dotenv(override=True)

class LLMGenerationStep(BaseStep):
    """
    Ultra-low latency grounded LLM generation powered by Groq API (llama-3.1-8b-instant).
    """
    name = "generation_llm"

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY") or settings.groq_api_key
        self.groq_client = None
        
        if self.api_key and self.api_key not in ("your_groq_api_key_here", "gsk_your_groq_api_key_here", ""):
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=self.api_key)
                logger.info(f"Initialized Groq Cloud LLM engine (Model: {settings.groq_model})")
            except Exception as e:
                logger.warning(f"Could not initialize Groq client: {e}")

    @retry_step(max_retries=2, base_delay=0.05, max_delay=0.2)
    def _call_groq(self, prompt: str) -> str:
        api_key = os.getenv("GROQ_API_KEY") or self.api_key
        if not self.groq_client:
            from groq import Groq
            self.groq_client = Groq(api_key=api_key)

        chat_completion = self.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_RAG_PROMPT},
                {"role": "user", "content": prompt}
            ],
            model=settings.groq_model,
            temperature=0.1,
            max_tokens=100
        )
        return chat_completion.choices[0].message.content.strip()

    def execute(self, input_data: dict) -> StepResult:
        question = input_data.get("transcript", "")
        context = input_data.get("context", "")

        if not question:
            return StepResult(success=False, error="No query supplied to LLM step.")

        full_prompt = format_user_prompt(question, context)

        # 1. Primary: Use Groq API (~60 - 90ms)
        api_key = os.getenv("GROQ_API_KEY") or self.api_key
        if api_key and api_key not in ("your_groq_api_key_here", "gsk_your_groq_api_key_here", ""):
            try:
                answer = self._call_groq(full_prompt)
                return StepResult(
                    success=True,
                    data={"answer": answer, "provider": "groq", "model": settings.groq_model}
                )
            except Exception as e:
                logger.warning(f"Groq API call error ({e}). Activating fallback...")

        # 2. Local Fallback
        lines = [l for l in context.split("\n") if l and not l.startswith("Passage")]
        concise = " ".join(lines)[:250] + "." if lines else context[:250]
        return StepResult(
            success=True,
            data={"answer": f"Grounded Answer: {concise}", "provider": "extractive_fallback"}
        )
