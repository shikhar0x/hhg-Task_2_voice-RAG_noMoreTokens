import time
import uuid
from typing import Any
from stt.engine import SpeechToTextStep
from retrieval.vector_store import VectorRetrievalStep
from guardrails.threshold_gate import GroundingGuardrailStep
from generation.llm import LLMGenerationStep
from infrastructure.metrics_db import MetricsDB
from config.logger import logger

# Layer 5: Hedge / Non-Answer phrases. When the model's own answer text signals it
# lacks grounded information, treat that as a refusal instead of leaking the
# non-answer through as an ANSWER (which would inflate false positives).
HEDGE_PHRASES = [
    "i don't have", "i do not have", "i don't have any", "i do not have any",
    "don't have information", "do not have information",
    "do not contain any information", "does not contain any information",
    "do not contain information", "does not contain information",
    "do not mention", "does not mention", "don't mention", "doesn't mention",
    "do not explicitly mention", "does not explicitly mention",
    "do not explicitly state", "does not explicitly state",
    "not directly mentioned", "not mentioned",
    "there is no information", "there is no information provided",
    "no information provided", "no information about",
    "unfortunately, the provided context", "unfortunately, the provided passages",
    "unable to provide", "unable to answer", "cannot provide", "cannot answer",
    "i'm unable", "i am unable", "we are unable",
    "do not specify", "does not specify", "not specified",
    "does not appear", "do not provide enough",
]

def is_hedge_answer(answer: str) -> bool:
    """Returns True if the model's answer text is itself a hedge / non-answer."""
    if not answer:
        return False
    lowered = answer.lower()
    return any(phrase in lowered for phrase in HEDGE_PHRASES)


class VoiceRAGOrchestrator:
    """Central pipeline harness running end-to-end Voice-RAG execution."""

    def __init__(self):
        self.stt = SpeechToTextStep()
        self.retrieval = VectorRetrievalStep()
        self.guardrail = GroundingGuardrailStep()
        self.generation = LLMGenerationStep()
        self.metrics_db = MetricsDB()

    def process(self, audio_path: str | None = None, text_override: str | None = None, mode: str | None = None) -> dict[str, Any]:
        query_id = str(uuid.uuid4())[:8]
        start_total = time.perf_counter()
        timings: dict[str, float] = {}

        if not mode:
            mode = "retrieval_only" if (text_override and not audio_path) else "end_to_end"

        # 1. STT (Speech-to-Text)
        stt_res = self.stt.run({"audio_path": audio_path, "text_override": text_override})
        timings["stt"] = stt_res.duration_ms
        if not stt_res.success:
            return {
                "query_id": query_id,
                "success": False,
                "error": stt_res.error,
                "stt_engine": "error",
                "timings": timings
            }

        transcript = stt_res.data.get("transcript", "")
        stt_engine = stt_res.data.get("engine", "stt")

        # 2. Retrieval (ChromaDB Vector Store)
        ret_res = self.retrieval.run({"transcript": transcript})
        timings["retrieval"] = ret_res.duration_ms
        if not ret_res.success:
            return {
                "query_id": query_id,
                "success": False,
                "error": ret_res.error,
                "transcript": transcript,
                "stt_engine": stt_engine,
                "timings": timings
            }

        # 3. Guardrail Gate (Layers 1-3: "Know when not to answer")
        guard_res = self.guardrail.run({
            "query": transcript,
            "retrieval_result": ret_res.data
        })
        timings["guardrail"] = guard_res.duration_ms

        if guard_res.refused:
            timings["generation"] = 0.0
            timings["hallucination_check"] = 0.0
            timings["hedge_check"] = 0.0
            timings["total"] = (time.perf_counter() - start_total) * 1000.0
            self.metrics_db.log(query_id, transcript, timings, refused=True, mode=mode)
            return {
                "query_id": query_id,
                "transcript": transcript,
                "answer": guard_res.data.get("answer"),
                "refused": True,
                "refusal_reason": guard_res.refusal_reason,
                "similarities": ret_res.data.get("similarities", []),
                "stt_engine": stt_engine,
                "timings": timings
            }

        # 4. LLM Generation (Groq Meta LLaMA-3.1)
        context = guard_res.data.get("valid_context", "")
        gen_res = self.generation.run({
            "transcript": transcript,
            "context": context
        })
        timings["generation"] = gen_res.duration_ms
        answer = gen_res.data.get("answer", "")

        # 5. Guardrail Layer 4: Post-Generation Hallucination & Faithfulness Check
        start_hallucination = time.perf_counter()
        is_faithful = self.guardrail.check_hallucination(answer, context)
        timings["hallucination_check"] = (time.perf_counter() - start_hallucination) * 1000.0

        if not is_faithful:
            timings["hedge_check"] = 0.0
            timings["total"] = (time.perf_counter() - start_total) * 1000.0
            refusal_reason = "Refusal: Generated answer failed post-generation grounding check against retrieved context."
            logger.warning(f"Guardrail Layer 4 Triggered: {refusal_reason}")
            fallback_answer = "I cannot provide this answer as it failed post-generation grounding verification against retrieved context."
            self.metrics_db.log(query_id, transcript, timings, refused=True, mode=mode)
            return {
                "query_id": query_id,
                "transcript": transcript,
                "answer": fallback_answer,
                "refused": True,
                "refusal_reason": refusal_reason,
                "similarities": ret_res.data.get("similarities", []),
                "stt_engine": stt_engine,
                "generation_provider": gen_res.data.get("provider", "unknown"),
                "timings": timings
            }

        # 6. Guardrail Layer 5: Hedge / Non-Answer Detection
        # Even when an answer passes numeric + lexical grounding, the model may itself
        # indicate it lacks grounded information ("I do not have any information about
        # ..."). Treat that as a refusal so it is logged/benchmarked as REFUSE.
        start_hedge = time.perf_counter()
        hedge_triggered = is_hedge_answer(answer)
        timings["hedge_check"] = (time.perf_counter() - start_hedge) * 1000.0
        timings["total"] = (time.perf_counter() - start_total) * 1000.0

        if hedge_triggered:
            hedge_reason = "Refusal: Model indicated insufficient grounded information to answer."
            logger.warning(f"Guardrail Layer 5 Triggered: {hedge_reason}")
            hedge_fallback = "I cannot provide this answer as the model indicated insufficient grounded information to answer from the retrieved context."
            self.metrics_db.log(query_id, transcript, timings, refused=True, mode=mode)
            return {
                "query_id": query_id,
                "transcript": transcript,
                "answer": hedge_fallback,
                "refused": True,
                "refusal_reason": hedge_reason,
                "similarities": ret_res.data.get("similarities", []),
                "stt_engine": stt_engine,
                "generation_provider": gen_res.data.get("provider", "unknown"),
                "timings": timings
            }

        self.metrics_db.log(query_id, transcript, timings, refused=False, mode=mode)

        return {
            "query_id": query_id,
            "transcript": transcript,
            "answer": answer,
            "refused": False,
            "similarities": ret_res.data.get("similarities", []),
            "retrieved_docs": ret_res.data.get("documents", []),
            "stt_engine": stt_engine,
            "generation_provider": gen_res.data.get("provider", "unknown"),
            "timings": timings
        }
