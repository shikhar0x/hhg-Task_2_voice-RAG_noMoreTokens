from config.settings import settings
from config.logger import logger
from harness.base import BaseStep, StepResult

class GroundingGuardrailStep(BaseStep):
    """
    Evaluates retrieval grounding confidence.
    Enforces 'Knowing when NOT to answer' by halting pipeline execution
    if context is missing or similarity score is below threshold.
    """
    name = "guardrail_grounding"

    def __init__(self, threshold: float | None = None):
        self.threshold = threshold if threshold is not None else settings.similarity_threshold

    def execute(self, input_data: dict) -> StepResult:
        retrieval_data = input_data.get("retrieval_result", {})
        docs = retrieval_data.get("documents", [])
        top_similarity = retrieval_data.get("top_similarity", 0.0)

        # Gate 1: No documents returned
        if not docs:
            reason = "Refusal: Zero relevant passages found in knowledge base."
            logger.info(f"Guardrail triggered: {reason}")
            return StepResult(
                success=True,
                refused=True,
                refusal_reason=reason,
                data={"answer": "I do not have sufficient information in the verified dataset to answer this question accurately."}
            )

        # Gate 2: Low confidence similarity cutoff
        if top_similarity < self.threshold:
            reason = f"Refusal: Top similarity score ({top_similarity:.3f}) is below strict confidence threshold ({self.threshold:.3f})."
            logger.info(f"Guardrail triggered: {reason}")
            return StepResult(
                success=True,
                refused=True,
                refusal_reason=reason,
                data={"answer": "This question is out of domain or lacks strong grounding in the provided dataset. Refusing to guess."}
            )

        # Context is validated
        valid_context = "\n\n".join([f"Passage {i+1}:\n{doc}" for i, doc in enumerate(docs)])
        return StepResult(
            success=True,
            refused=False,
            data={"valid_context": valid_context, "top_similarity": top_similarity}
        )
