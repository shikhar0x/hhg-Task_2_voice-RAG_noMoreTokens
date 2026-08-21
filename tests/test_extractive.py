"""Sprint 0 contract: extractive exists before generation and never depends on it."""
import pytest

from harness.base import StepResult
from harness.orchestrator import VoiceRAGOrchestrator
from retrieval.extractive import extract_answer


CORPUS_DOC = (
    "A corporation is a company or group of people authorized to act as a "
    "single entity and recognized as such in law."
)


def _orch(mocker, *, docs=None, generated=None):
    orch = VoiceRAGOrchestrator()
    mocker.patch.object(orch.stt, "run", return_value=StepResult(
        success=True, data={"transcript": "what is a corporation?", "engine": "text_input"}
    ))
    mocker.patch.object(orch.retrieval, "run", return_value=StepResult(
        success=True,
        data={
            "documents": docs or [CORPUS_DOC],
            "similarities": [0.85],
            "top_similarity": 0.85,
        },
    ))
    if generated is not None:
        mocker.patch.object(orch.generation, "run", return_value=StepResult(
            success=True, data={"answer": generated, "provider": "Groq"}
        ))
    return orch


def test_extract_answer_picks_overlapping_sentence():
    docs = [
        "Unrelated preamble about shipping lanes. "
        "A corporation is a company authorized to act as a single entity. "
        "The weather in Goa is humid in August."
    ]
    ans = extract_answer("what is a corporation?", docs)
    assert "corporation" in ans.text.lower()
    assert "authorized" in ans.text.lower()
    assert ans.support > 0
    assert ans.took_ms >= 0.0


def test_extract_skips_dictionary_header_for_definition():
    """Regression: the live Sprint 0 smoke test returned the MW header."""
    docs = [
        "Examples of corporation in a Sentence. "
        "He works as a consultant for several large corporations. "
        "a substantial corporation that showed that he was a sucker for all-you-can-eat buffets.",
        "1: a government-owned corporation (as a utility or railroad) engaged in a profit-making enterprise",
        "A company is incorporated in a specific nation, often within the bounds of a smaller subset of that nation, such as a state or province. "
        "The corporation is then governed by the laws of incorporation in that state.",
    ]
    sims = [0.381, 0.378, 0.331]
    ans = extract_answer("what is a corporation?", docs, similarities=sims)
    assert "examples of corporation" not in ans.text.lower()
    assert "incorporated" in ans.text.lower() or "governed" in ans.text.lower()


def test_generate_false_never_calls_llm(mocker):
    orch = _orch(mocker)
    orch.generation.run = mocker.Mock(side_effect=AssertionError("LLM must not run"))

    res = orch.process(text_override="what is a corporation?", generate=False)

    assert res["answer_source"] == "extractive"
    assert res["refused"] is False
    assert "corporation" in (res["answer"] or "").lower()
    assert res["extractive_answer"] == res["answer"]
    assert res["generated_answer"] == ""
    assert res["timings"]["generation"] == 0.0
    assert "fast_path" in res["timings"]
    assert res["fast_path_ms"] == res["timings"]["fast_path"]
    assert res["budget_ms"] == 200.0
    assert res["within_budget"] is True
    assert res["sources"]


def test_weak_span_abstains_without_llm(mocker):
    """L3 can pass on a high cosine to the wrong topic; the span gate must still abstain."""
    orch = _orch(
        mocker,
        docs=["The weather in Goa is humid in August and the beaches are crowded with tourists."],
    )
    orch.generation.run = mocker.Mock(side_effect=AssertionError("LLM must not run on abstain"))
    res = orch.process(text_override="what is a corporation?", generate=True)
    assert res["answer_source"] == "abstain"
    assert res["refused"] is True
    assert res["timings"]["generation"] == 0.0
    assert "abstain:" in (res.get("refusal_reason") or "")


def test_grounding_verdict_rejects_empty_and_offtopic():
    from retrieval.extractive import ExtractiveAnswer, extract_answer, grounding_verdict
    ok, reason, _ = grounding_verdict("what is a corporation?", ExtractiveAnswer("", 0.0, "", 0, 0.0))
    assert ok is False
    assert reason == "empty_span"
    weak = extract_answer(
        "what is a corporation?",
        ["Monsoon rains flood the Mandovi every July."],
    )
    ok, reason, cov = grounding_verdict("what is a corporation?", weak)
    assert ok is False


def test_failed_generation_keeps_extractive(mocker):
    orch = _orch(
        mocker,
        generated="Quantum computers rely on Shor algorithm and RSA 2048 keys.",
    )
    res = orch.process(text_override="what is a corporation?", generate=True)

    assert res["refused"] is False
    assert res["answer_source"] == "extractive"
    assert "corporation" in (res["answer"] or "").lower()
    assert "generation_rejected" in (res.get("refusal_reason") or "")
    assert res["extractive_answer"]
    assert res["generated_answer"]


def test_faithful_generation_replaces_extractive(mocker):
    polished = "A corporation is a company recognized as a single legal entity."
    orch = _orch(mocker, generated=polished)
    res = orch.process(text_override="what is a corporation?", generate=True)

    assert res["refused"] is False
    assert res["answer_source"] == "generated"
    assert res["answer"] == polished
    assert res["extractive_answer"]
    assert res["generated_answer"] == polished
