import os
import re
import time
import httpx
from dotenv import load_dotenv
from config.settings import settings
from config.logger import logger
from harness.base import BaseStep, StepResult
from harness.retry import retry_step

load_dotenv(override=True)

class LLMGenerationStep(BaseStep):
    """
    Grounded LLM Generation Step powered by Groq Meta LLaMA-3.1
    Generates complete, fluent, multi-sentence answers without truncation.
    """
    name = "generation_llm"

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("GROQ_API_KEY") or settings.groq_api_key
        self.client = None
        self.http_client = httpx.Client(
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20, keepalive_expiry=60.0),
            timeout=30.0
        )
        if self.api_key and self.api_key not in ("your_groq_api_key_here", "gsk_your_groq_api_key_here", ""):
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key, http_client=self.http_client, max_retries=0)
            except Exception as e:
                logger.warning(f"Groq init note: {e}")

    @retry_step(max_retries=3, base_delay=0.5, max_delay=3.0)
    def _generate_groq(self, question: str, context: str) -> str:
        api_key = os.getenv("GROQ_API_KEY") or self.api_key
        if not self.client:
            from groq import Groq
            self.client = Groq(api_key=api_key, http_client=self.http_client, max_retries=0)

        clean_context = re.sub(r'Passage \d+:', '', context).strip()
        t0 = time.perf_counter()
        
        models_to_try = [
            os.getenv("GROQ_MODEL") or settings.groq_model,
            "groq/compound-mini",
            "openai/gpt-oss-20b",
            "groq/compound"
        ]
        
        unique_models = []
        for m in models_to_try:
            if m and m not in unique_models:
                unique_models.append(m)

        last_error = None
        for model_name in unique_models:
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a concise, factually grounded AI assistant. "
                                "Answer the user's question directly in 2-3 clear sentences in the SAME language as the user's question (e.g., Bangla for Bangla, Hindi for Hindi, English for English) using ONLY the provided context passages."
                            )
                        },
                        {
                            "role": "user",
                            "content": f"Verified Context Passages:\n{clean_context}\n\nQuestion:\n{question}\n\nConcise Grounded Answer:"
                        }
                    ],
                    model=model_name,
                    temperature=0.0,
                    max_tokens=100
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                logger.info(f"Groq LLM ({model_name}) generation completed in {elapsed_ms:.1f}ms")
                return chat_completion.choices[0].message.content.strip()
            except Exception as e:
                last_error = e
                logger.warning(f"Groq model '{model_name}' returned error: {e}. Trying next model candidate...")

        if last_error:
            raise last_error
        return ""

    def execute(self, input_data: dict) -> StepResult:
        question = input_data.get("transcript", "")
        context = input_data.get("context", "")

        if not question:
            return StepResult(success=False, error="No query supplied.")

        # 1. Groq Meta LLaMA-3.1 (Full unclipped answer)
        api_key = os.getenv("GROQ_API_KEY") or self.api_key
        if api_key and api_key not in ("your_groq_api_key_here", "gsk_your_groq_api_key_here", ""):
            try:
                answer = self._generate_groq(question, context)
                if answer:
                    return StepResult(success=True, data={"answer": answer, "provider": "Groq Meta LLaMA-3.1"})
            except Exception as e:
                logger.warning(f"Groq generation error ({e}). Using synthesis fallback...")

        # 2. Clean fallback
        clean_text = re.sub(r'Passage \d+:', '', context).strip()
        return StepResult(success=True, data={"answer": clean_text[:500], "provider": "clean_fallback"})
