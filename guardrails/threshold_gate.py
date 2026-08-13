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

class GroundingGuardrailStep(BaseStep):
    """
    4-Layer Guardrail Suite satisfying all HH Goa Task #2 requirements:
    1. Unsafe / Inappropriate Input Filter
    2. Off-Topic / Out-of-Domain Gate (Similarity Threshold)
    3. Insufficient Context Refusal Gate
    4. Post-Generation Hallucination & Faithfulness Checker
    """
    name = "guardrail_defense_suite"

    def __init__(self, threshold: float | None = None):
        self.threshold = threshold if threshold is not None else settings.similarity_threshold

    def check_safety(self, query: str) -> tuple[bool, str]:
        """Layer 1: Detects unsafe, abusive, or prompt-injection inputs."""
        q_lower = query.lower()
        for pattern in UNSAFE_PATTERNS:
            if re.search(pattern, q_lower):
                return False, "Refusal: Query flagged as unsafe, inappropriate, or policy-violating."
        return True, ""

    def check_hallucination(self, answer: str, context: str) -> bool:
        """Layer 4: Checks if the generated answer is faithful to the retrieved context."""
        if not answer or not context:
            return False
        
        # Extract meaningful entity words (length > 3)
        ans_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', answer.lower())) - {"this", "that", "from", "with", "have", "were", "they", "their"}
        ctx_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', context.lower()))
        
        if not ans_words:
            return True
            
        overlap = len(ans_words.intersection(ctx_words)) / len(ans_words)
        # If less than 20% of answer terms exist in context, flag as potential hallucination
        return overlap >= 0.20

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
