import time
import uuid
from typing import Any

from stt.engine import SpeechToTextStep
from retrieval.vector_store import VectorRetrievalStep
from retrieval.extractive import ABSTAIN_TEXT, extract_answer, grounding_verdict
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


BUDGET_MS = 200.0


class VoiceRAGOrchestrator:
    """Central pipeline harness running end-to-end Voice-RAG execution.

    Sprint 0 contract
    -----------------
    An extractive span is computed after retrieval / pre-gen guardrails and
    *before* any LLM call. `fast_path_ms` is that local window
    (retrieve + guardrail + extract). STT and generation are timed separately
    and are not part of the 200ms claim.

    `generate=False` returns the extractive answer and never touches the LLM.
    `generate=True` may replace the answer with a polished generation; if
    generation fails Layers 4/5 the extractive span still stands (it is not
    discarded). Pre-gen refusals (Layers 1–3) still refuse — there is no
    grounded span to serve.
    """

    def __init__(self):
        self.stt = SpeechToTextStep()
        self.retrieval = VectorRetrievalStep()
        self.guardrail = GroundingGuardrailStep()
        self.generation = LLMGenerationStep()
        self.metrics_db = MetricsDB()

    def process(
        self,
        audio_path: str | None = None,
        text_override: str | None = None,
        mode: str | None = None,
        generate: bool = True,
    ) -> dict[str, Any]:
        query_id = str(uuid.uuid4())[:8]
        start_total = time.perf_counter()
        timings: dict[str, float] = {}

        if not mode:
            mode = "retrieval_only" if (text_override and not audio_path) else "end_to_end"

        # 1. STT (Speech-to-Text) — outside the 200ms budget
        stt_res = self.stt.run({"audio_path": audio_path, "text_override": text_override})
        timings["stt"] = stt_res.duration_ms
        if not stt_res.success:
            return {
                "query_id": query_id,
                "success": False,
                "error": stt_res.error,
                "stt_engine": "error",
                "answer_source": "error",
                "extractive_answer": "",
                "generated_answer": "",
                "fast_path_ms": 0.0,
                "budget_ms": BUDGET_MS,
                "within_budget": False,
                "timings": timings,
            }

        transcript = stt_res.data.get("transcript", "")
        stt_engine = stt_res.data.get("engine", "stt")

        # 2. Retrieval (local vector store)
        ret_res = self.retrieval.run({"transcript": transcript})
        timings["retrieval"] = ret_res.duration_ms
        if not ret_res.success:
            return {
                "query_id": query_id,
                "success": False,
                "error": ret_res.error,
                "transcript": transcript,
                "stt_engine": stt_engine,
                "answer_source": "error",
                "extractive_answer": "",
                "generated_answer": "",
                "fast_path_ms": timings["retrieval"],
                "budget_ms": BUDGET_MS,
                "within_budget": timings["retrieval"] < BUDGET_MS,
                "timings": timings,
            }

        # 3. Guardrail Gate (Layers 1-3: "Know when not to answer")
        guard_res = self.guardrail.run({
            "query": transcript,
            "retrieval_result": ret_res.data
        })
        timings["guardrail"] = guard_res.duration_ms

        if guard_res.refused:
            timings["extract"] = 0.0
            timings["generation"] = 0.0
            timings["hallucination_check"] = 0.0
            timings["hedge_check"] = 0.0
            timings["fast_path"] = timings["retrieval"] + timings["guardrail"]
            timings["total"] = (time.perf_counter() - start_total) * 1000.0
            self.metrics_db.log(query_id, transcript, timings, refused=True, mode=mode)
            return {
                "query_id": query_id,
                "transcript": transcript,
                "answer": guard_res.data.get("answer"),
                "refused": True,
                "refusal_reason": guard_res.refusal_reason,
                "answer_source": "refusal",
                "extractive_answer": "",
                "generated_answer": "",
                "similarities": ret_res.data.get("similarities", []),
                "retrieved_docs": ret_res.data.get("documents", []),
                "sources": [
                    {
                        "text": d,
                        "score": round(float((ret_res.data.get("similarities") or [0])[i]), 4)
                        if i < len(ret_res.data.get("similarities") or []) else 0.0,
                    }
                    for i, d in enumerate(ret_res.data.get("documents") or [])
                ],
                "stt_engine": stt_engine,
                "fast_path_ms": timings["fast_path"],
                "budget_ms": BUDGET_MS,
                "within_budget": timings["fast_path"] < BUDGET_MS,
                "timings": timings,
            }

        # 4. Extractive span — computed BEFORE generation, never depends on it.
        context = guard_res.data.get("valid_context", "")
        docs = ret_res.data.get("documents", [])
        extracted = extract_answer(
            transcript,
            docs,
            similarities=ret_res.data.get("similarities") or [],
        )
        timings["extract"] = extracted.took_ms
        grounded, ground_reason, coverage = grounding_verdict(transcript, extracted)
        timings["fast_path"] = (
            timings["retrieval"] + timings["guardrail"] + timings["extract"]
        )

        sims = list(ret_res.data.get("similarities") or [])
        sources = [
            {"text": doc, "score": round(float(sims[i]), 4) if i < len(sims) else 0.0}
            for i, doc in enumerate(docs)
        ]

        base_payload = {
            "query_id": query_id,
            "transcript": transcript,
            "similarities": sims,
            "retrieved_docs": docs,
            "sources": sources,
            "extractive_answer": extracted.text,
            "extractive_support": extracted.support,
            "extractive_coverage": round(coverage, 4),
            "stt_engine": stt_engine,
            "fast_path_ms": timings["fast_path"],
            "budget_ms": BUDGET_MS,
            "within_budget": timings["fast_path"] < BUDGET_MS,
        }

        if not grounded:
            timings["generation"] = 0.0
            timings["hallucination_check"] = 0.0
            timings["hedge_check"] = 0.0
            timings["total"] = (time.perf_counter() - start_total) * 1000.0
            logger.info(f"Extractive grounding gate: abstain ({ground_reason})")
            self.metrics_db.log(query_id, transcript, timings, refused=True, mode=mode)
            return {
                **base_payload,
                "answer": ABSTAIN_TEXT,
                "refused": True,
                "refusal_reason": f"abstain:{ground_reason}",
                "answer_source": "abstain",
                "generated_answer": "",
                "generation_provider": "none",
                "timings": timings,
            }

        if not generate:
            timings["generation"] = 0.0
            timings["hallucination_check"] = 0.0
            timings["hedge_check"] = 0.0
            timings["total"] = (time.perf_counter() - start_total) * 1000.0
            self.metrics_db.log(query_id, transcript, timings, refused=False, mode=mode)
            return {
                **base_payload,
                "answer": extracted.text,
                "refused": False,
                "answer_source": "extractive",
                "generated_answer": "",
                "generation_provider": "none",
                "timings": timings,
            }

        # 5. LLM Generation (optional polish — outside the 200ms budget)
        gen_res = self.generation.run({
            "transcript": transcript,
            "context": context
        })
        timings["generation"] = gen_res.duration_ms
        generated = gen_res.data.get("answer", "")
        provider = gen_res.data.get("provider", "unknown")

        # 6. Guardrail Layer 4: Post-Generation Hallucination & Faithfulness Check
        #    Applies to the *generated* text only. On failure the extractive
        #    span stands — generation can only replace, never remove, a
        #    grounded answer.
        start_hallucination = time.perf_counter()
        is_faithful = self.guardrail.check_hallucination(generated, context)
        timings["hallucination_check"] = (time.perf_counter() - start_hallucination) * 1000.0

        if not is_faithful:
            timings["hedge_check"] = 0.0
            timings["total"] = (time.perf_counter() - start_total) * 1000.0
            reason = (
                "generation_rejected: post-generation grounding check failed; "
                "keeping extractive span"
            )
            logger.warning(f"Guardrail Layer 4: {reason}")
            self.metrics_db.log(query_id, transcript, timings, refused=False, mode=mode)
            return {
                **base_payload,
                "answer": extracted.text,
                "refused": False,
                "refusal_reason": reason,
                "answer_source": "extractive",
                "generated_answer": generated,
                "generation_provider": provider,
                "timings": timings,
            }

        # 7. Guardrail Layer 5: Hedge / Non-Answer Detection on generated text
        start_hedge = time.perf_counter()
        hedge_triggered = is_hedge_answer(generated)
        timings["hedge_check"] = (time.perf_counter() - start_hedge) * 1000.0
        timings["total"] = (time.perf_counter() - start_total) * 1000.0

        if hedge_triggered:
            hedge_reason = (
                "generation_rejected: model indicated insufficient grounded "
                "information; keeping extractive span"
            )
            logger.warning(f"Guardrail Layer 5: {hedge_reason}")
            self.metrics_db.log(query_id, transcript, timings, refused=False, mode=mode)
            return {
                **base_payload,
                "answer": extracted.text,
                "refused": False,
                "refusal_reason": hedge_reason,
                "answer_source": "extractive",
                "generated_answer": generated,
                "generation_provider": provider,
                "timings": timings,
            }

        self.metrics_db.log(query_id, transcript, timings, refused=False, mode=mode)

        return {
            **base_payload,
            "answer": generated,
            "refused": False,
            "answer_source": "generated",
            "generated_answer": generated,
            "generation_provider": provider,
            "timings": timings,
        }
