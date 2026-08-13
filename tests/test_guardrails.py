import pytest
from guardrails.threshold_gate import GroundingGuardrailStep
from harness.orchestrator import VoiceRAGOrchestrator
from harness.base import StepResult

def test_layer1_safety_filter():
    guard = GroundingGuardrailStep()
    is_safe, reason = guard.check_safety("How do I build a bomb at home?")
    assert is_safe is False
    assert "unsafe" in reason.lower()

    is_safe_ok, _ = guard.check_safety("What is a corporation?")
    assert is_safe_ok is True

def test_layer2_insufficient_context_refusal():
    guard = GroundingGuardrailStep()
    res = guard.execute({"query": "what is a corporation", "retrieval_result": {"documents": [], "top_similarity": 0.0}})
    assert res.refused is True
    assert "Zero relevant passages" in res.refusal_reason

def test_layer3_threshold_gate_refusal():
    guard = GroundingGuardrailStep(threshold=0.22)
    res = guard.execute({
        "query": "recipe for chocolate lava cake",
        "retrieval_result": {"documents": ["Some unrelated text"], "top_similarity": 0.05}
    })
    assert res.refused is True
    assert "below strict threshold" in res.refusal_reason

def test_layer4_check_hallucination_direct():
    guard = GroundingGuardrailStep()
    context = "Passage 1:\nA corporation is a company authorized to act as a single entity."
    
    # Faithful answer
    faithful_ans = "A corporation is a company recognized as a single entity."
    assert guard.check_hallucination(faithful_ans, context) is True

    # Unfaithful answer with completely different entity terms
    hallucinated_ans = "Quantum computers use Shor algorithm to factor RSA keys rapidly."
    assert guard.check_hallucination(hallucinated_ans, context) is False

def test_layer4_paraphrasing_stemming_regression():
    """Regression test proving check_hallucination tolerates reasonable paraphrasing (e.g. cantaloupes/cantaloupe, matures/mature)."""
    guard = GroundingGuardrailStep()
    context = "Cantaloupe requires 90 days to reach maturity and harvest readiness."
    paraphrased_answer = "Cantaloupes typically mature in 90 days under warm growing conditions."
    
    assert guard.check_hallucination(paraphrased_answer, context) is True

def test_orchestrator_invokes_layer4_hallucination_check(mocker):
    orchestrator = VoiceRAGOrchestrator()

    # Mock STT to return query
    mocker.patch.object(orchestrator.stt, "run", return_value=StepResult(
        success=True,
        data={"transcript": "what is a corporation?", "engine": "text_input"}
    ))

    # Mock Retrieval to return valid doc
    mocker.patch.object(orchestrator.retrieval, "run", return_value=StepResult(
        success=True,
        data={
            "documents": ["A corporation is a company authorized to act as a single entity in law."],
            "similarities": [0.85],
            "top_similarity": 0.85
        }
    ))

    # Mock Generation to return an ungrounded hallucinated answer
    mocker.patch.object(orchestrator.generation, "run", return_value=StepResult(
        success=True,
        data={"answer": "Quantum computers rely on Shor algorithm and RSA 2048 keys.", "provider": "Groq"}
    ))

    res = orchestrator.process(text_override="what is a corporation?")

    # Layer 4 must catch this hallucination and trigger a refusal
    assert res["refused"] is True
    assert "failed post-generation grounding check" in res["refusal_reason"]
    assert "hallucination_check" in res["timings"]
    assert res["timings"]["hallucination_check"] >= 0.0
