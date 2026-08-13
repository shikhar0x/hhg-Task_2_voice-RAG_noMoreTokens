import time
import uuid
from typing import Any
from stt.engine import SpeechToTextStep
from retrieval.vector_store import VectorRetrievalStep
from guardrails.threshold_gate import GroundingGuardrailStep
from generation.llm import LLMGenerationStep
from infrastructure.metrics_db import MetricsDB
from config.logger import logger

class VoiceRAGOrchestrator:
    """Central pipeline harness running end-to-end Voice-RAG execution."""

    def __init__(self):
        self.stt = SpeechToTextStep()
        self.retrieval = VectorRetrievalStep()
        self.guardrail = GroundingGuardrailStep()
        self.generation = LLMGenerationStep()
        self.metrics_db = MetricsDB()

    def process(self, audio_path: str | None = None, text_override: str | None = None) -> dict[str, Any]:
        query_id = str(uuid.uuid4())[:8]
        start_total = time.perf_counter()
        timings: dict[str, float] = {}

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

        # 3. Guardrail Gate ("Know when not to answer")
        guard_res = self.guardrail.run({
            "query": transcript,
            "retrieval_result": ret_res.data
        })
        timings["guardrail"] = guard_res.duration_ms

        if guard_res.refused:
            timings["generation"] = 0.0
            timings["total"] = (time.perf_counter() - start_total) * 1000.0
            self.metrics_db.log(query_id, transcript, timings, refused=True)
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
        gen_res = self.generation.run({
            "transcript": transcript,
            "context": guard_res.data.get("valid_context", "")
        })
        timings["generation"] = gen_res.duration_ms
        timings["total"] = (time.perf_counter() - start_total) * 1000.0

        answer = gen_res.data.get("answer", "")
        self.metrics_db.log(query_id, transcript, timings, refused=False)

        return {
            "query_id": query_id,
            "transcript": transcript,
            "answer": answer,
            "refused": False,
            "similarities": ret_res.data.get("similarities", []),
            "retrieved_docs": ret_res.data.get("documents", []),
            "stt_engine": stt_engine,
            "timings": timings
        }
