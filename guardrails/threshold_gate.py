import re
from typing import Any
from config.settings import settings
from config.logger import logger
from harness.base import BaseStep, StepResult

# Layer 1: Safety & Moderation Blacklist
UNSAFE_PATTERNS = [
    r'\b(hack|bypass|jailbreak|exploit|malware|ransomware|ddos)\b',
    r'\b(bomb|weapon|explosive|poison|terrorist|attack)\b',
    r'\b(ignore previous instructions|system prompt|disregard rules)\b'
]

def stem_word(w: str) -> str:
    """
    Lightweight English suffix-stripping stemmer.
    Normalizes common inflections (-s, -es, -ed, -ing, -ly, -ity, -ies, trailing -e)
    so paraphrases (e.g. 'cantaloupes'/'cantaloupe', 'matures'/'maturity')
    are recognized as equivalent terms during post-generation grounding checks.
    """
    w = w.lower()
    if w.endswith('ity') and len(w) > 5:
        w = w[:-3] + 'e'
    elif w.endswith('ies') and len(w) > 5:
        w = w[:-3] + 'y'
    elif w.endswith('ing') and len(w) > 5:
        w = w[:-3]
    elif w.endswith('ed') and len(w) > 4:
        w = w[:-2]
    elif w.endswith('ly') and len(w) > 4:
        w = w[:-2]
    elif w.endswith('s') and not w.endswith('ss') and len(w) > 3:
        w = w[:-1]
    if w.endswith('e') and len(w) > 4:
        w = w[:-1]
    return w

class GroundingGuardrailStep(BaseStep):
    """
    4-Layer Guardrail Suite satisfying all HH Goa Task #2 requirements:
    1. Unsafe / Inappropriate Input Filter
    2. Off-Topic / Out-of-Domain Gate (Similarity Threshold)
    3. Insufficient Context Refusal Gate
    4. Post-Generation Hallucination & Faithfulness Checker
    """
    name = "guardrail_defense_suite"

    def __init__(self, threshold: float | None = None, hallucination_threshold: float = 0.20):
        self.threshold = threshold if threshold is not None else settings.similarity_threshold
        self.hallucination_threshold = hallucination_threshold

    def check_safety(self, query: str) -> tuple[bool, str]:
        """Layer 1: Detects unsafe, abusive, or prompt-injection inputs."""
        q_lower = query.lower()
        for pattern in UNSAFE_PATTERNS:
            if re.search(pattern, q_lower):
                return False, "Refusal: Query flagged as unsafe, inappropriate, or policy-violating."
        return True, ""

    def check_hallucination(self, answer: str, context: str) -> bool:
        """
        Layer 4: Checks if the generated answer is faithful to the retrieved context.

        Two sub-checks:
          (1) NUMERIC FABRICATION — every number token in the answer must also appear
              in the retrieved context. A specific figure absent from the context is a
              hallucination even when the surrounding words overlap heavily (e.g. an
              eagle "30-40 mph" answer over an Amtrak passage, or a "100-120 days"
              answer over a passage that only states honeydew "80 days"). A strictly
              grounded RAG system must derive every quantitative claim from the
              retrieved passages, never from parametric knowledge.
          (2) LEXICAL OVERLAP (stemmed) — the fraction of answer content words present
              in the context must meet hallucination_threshold (default 0.20), so
              reasonable paraphrases are tolerated.

        NOTE: a capitalized "proper-noun" check was evaluated and deliberately REMOVED.
        It was too crude for natural LLM prose — it flagged sentence-initial discourse
        markers ("However", "Additionally", "Unfortunately", "First") as fabricated
        proper nouns, producing false refusals on grounded answers. Catching semantic
        fabrications such as a wrong term definition (e.g. "amidships" = "bottom front"
        vs the passage's "midship section") or off-the-cuff genus names requires an
        LLM-as-judge entailment layer, which is deferred.
        """
        if not answer or not context:
            return False

        # --- (1) Numeric fabrication: every number in the answer must be grounded ---
        ans_nums = set(re.findall(r'\b\d[\d,.\-]*\b', answer.lower()))
        ctx_nums = set(re.findall(r'\b\d[\d,.\-]*\b', context.lower()))
        if ans_nums - ctx_nums:
            logger.info(
                f"Guardrail Layer 4: numeric fabrication detected "
                f"(invented numbers not in context: {sorted(ans_nums - ctx_nums)})."
            )
            return False

        # --- (2) Stemmed lexical overlap ---
        ans_raw = set(re.findall(r'\b[a-zA-Z]{4,}\b', answer.lower())) - {
            "this", "that", "from", "with", "have", "were", "they", "their",
            "about", "would", "which", "there", "these", "where"
        }
        ctx_raw = set(re.findall(r'\b[a-zA-Z]{4,}\b', context.lower()))

        if not ans_raw:
            return True

        ans_words = {stem_word(w) for w in ans_raw}
        ctx_words = {stem_word(w) for w in ctx_raw}

        overlap = len(ans_words.intersection(ctx_words)) / len(ans_words)
        return overlap >= self.hallucination_threshold

    def execute(self, input_data: dict[str, Any]) -> StepResult:
        query = input_data.get("query", "")
        retrieval_data = input_data.get("retrieval_result", {})
        docs = retrieval_data.get("documents", [])
        top_similarity = retrieval_data.get("top_similarity", 0.0)

        # ─── LAYER 1: UNSAFE / INAPPROPRIATE INPUT FILTER ───
        if query:
            is_safe, safety_reason = self.check_safety(query)
            if not is_safe:
                logger.warning(f"Guardrail Layer 1 Triggered: {safety_reason}")
                return StepResult(
                    success=True,
                    refused=True,
                    refusal_reason=safety_reason,
                    data={"answer": "I cannot answer this query as it violates safety guidelines or requests inappropriate content."}
                )

        # ─── LAYER 2: INSUFFICIENT CONTEXT CHECK ───
        if not docs:
            reason = "Refusal: Zero relevant passages found in the knowledge base."
            logger.info(f"Guardrail Layer 2 Triggered: {reason}")
            return StepResult(
                success=True,
                refused=True,
                refusal_reason=reason,
                data={"answer": "I do not have sufficient verified context in the dataset to answer this question accurately."}
            )

        # ─── LAYER 3: OFF-TOPIC CONFIDENCE THRESHOLD GATE ───
        if top_similarity < self.threshold:
            reason = f"Refusal: Similarity score ({top_similarity:.3f}) below strict threshold ({self.threshold:.3f})."
            logger.info(f"Guardrail Layer 3 Triggered: {reason}")
            return StepResult(
                success=True,
                refused=True,
                refusal_reason=reason,
                data={"answer": "This question is out of domain or lacks strong grounding in the provided dataset. Refusing to guess to prevent hallucination."}
            )

        # Validated Context
        valid_context = "\n\n".join([f"Passage {i+1}:\n{doc}" for i, doc in enumerate(docs)])
        return StepResult(
            success=True,
            refused=False,
            data={
                "valid_context": valid_context,
                "top_similarity": top_similarity,
                "guardrail_status": "Passed (Safety, Grounding & Context Verified)"
            }
        )
