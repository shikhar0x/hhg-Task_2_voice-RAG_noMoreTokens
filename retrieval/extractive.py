"""Sprint 0.5: lexical extractive answer across the retrieved set.

Still no embeddings and no HTTP. The Sprint 0 scorer only looked at the top
hit, so a dictionary header ("Examples of corporation in a Sentence.") beat
the actual definition sitting in hit #3. This version scores every sentence
in the retrieved window and down-weights usage-example passages.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

_SENT_END = re.compile(r"(?<=[।॥?!])\s+|(?<=[.!?])\s+")
_TOKEN = re.compile(r"[\u0900-\u097F\w]+")

_QUERY_STOP = {
    "what", "whats", "when", "where", "which", "who", "whom", "why", "how",
    "does", "did", "doing", "done", "the", "and", "for", "with", "from",
    "into", "that", "this", "your", "their", "there", "have", "has", "had",
    "was", "were", "are", "is", "been", "being", "will", "would", "could",
    "should", "can", "may", "many", "much", "long", "time", "define",
    "definition", "meaning", "mean", "explain", "about",
    "क्या", "कौन", "कहाँ", "कब", "कैसे", "क्यों", "है", "हैं", "का", "की", "के",
}

_BOILERPLATE = re.compile(
    r"(?ix)"
    r"^(examples?\s+of\b.+\bin a sentence\b)"
    r"|^(see also|related (?:words|terms)|synonyms?|antonyms?)\b"
    r"|^(from wikipedia|retrieved from|copyright)\b"
    r"|\bin a sentence\s*:?\s*$"
)

_EXAMPLE_PASSAGE = re.compile(
    r"(?ix)^(examples?\s+of\b)|\bin a sentence\b"
)

_EXAMPLE_SENT = re.compile(
    r"(?ix)^(he|she|they|i|we)\b|\bfor example\b|\be\.g\.|\bsucker\b"
)

_DEFINITIONAL = re.compile(
    r"(?ix)\b("
    r"is a|is an|are a|are an|refers to|means|defined as|known as|"
    r"consists of|is incorporated|was incorporated|"
    r"है एक|को कहते हैं"
    r")\b"
)

_DEF_QUERY = re.compile(
    r"(?ix)^(what(?:'s| is| are)\b)|(\bdefine\b)|(\bmeaning of\b)|(\bdefinition\b)|"
    r"(क्या है)|(\bम्हणजे काय\b)"
)

_SUFFIXES = (
    "ations", "ation", "ated", "ates", "ate",
    "ions", "ion", "ings", "ing", "ers", "er",
    "ies", "ied", "ed", "es", "s",
)


@dataclass(slots=True)
class ExtractiveAnswer:
    text: str
    support: float
    source_passage: str
    n_candidates: int
    took_ms: float

    @property
    def is_empty(self) -> bool:
        return not (self.text or "").strip()


def _tokens(text: str) -> set[str]:
    return {w.lower() for w in _TOKEN.findall(text or "") if len(w) > 2}


def _stem(w: str) -> str:
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def _related(q_terms: set[str], word: str) -> bool:
    """True if `word` is the query term or a morphological cousin.

    'corporation' must match 'incorporated' / 'corporate', otherwise the
    real definition ('A company is incorporated…') scores zero overlap.
    """
    wl = word.lower()
    if wl in q_terms:
        return True
    wstem = _stem(wl)
    for q in q_terms:
        if q == wl or _stem(q) == wstem:
            return True
        if len(q) >= 5 and (q in wl or wl in q or _stem(q) in wl):
            return True
    return False


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_END.split(text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _query_terms(query: str) -> set[str]:
    return {t for t in _tokens(query) if t not in _QUERY_STOP}


def _score_sentence(
    sent: str,
    q_terms: set[str],
    rank: int,
    sim: float,
    *,
    example_passage: bool,
    definition_query: bool,
) -> float:
    if len(sent) < 40:
        return -1.0
    if _BOILERPLATE.search(sent):
        return -1.0

    s_terms = _tokens(sent)
    if not s_terms:
        return -1.0

    hit = {t for t in s_terms if _related(q_terms, t)} if q_terms else set()
    overlap = (len(hit) / len(q_terms)) if q_terms else 0.0
    if overlap <= 0.0 and q_terms:
        return -1.0

    length = 1.0 if 60 <= len(sent) <= 420 else 0.72
    defin = _DEFINITIONAL.search(sent) is not None
    defin_w = 1.85 if defin else 1.0
    rank_w = 1.0 / (1.0 + 0.10 * rank)
    sim_w = 0.55 + 0.45 * max(0.0, min(1.0, sim))

    score = overlap * length * defin_w * rank_w * sim_w

    if example_passage:
        score *= 0.22
    if _EXAMPLE_SENT.search(sent):
        score *= 0.25
    if definition_query and not defin:
        score *= 0.45
    # Numbered dictionary senses ("1: a government-owned…") are definitions of
    # a *subtype*, not the asked-for headword. Keep them, but behind a general
    # definition if one exists in the window.
    if re.match(r"^\d+\s*:", sent.strip()):
        score *= 0.70

    return score


# Dual gate for "do we have a span worth serving?"
# Support is overlap × definitional/rank boosts (can exceed 1.0).
# Coverage is the fraction of salient query terms attested in the span.
MIN_SUPPORT = 0.20
MIN_COVERAGE = 0.25

ABSTAIN_TEXT = (
    "I don't have enough grounded information in the indexed passages to answer that."
)


def query_coverage(query: str, text: str) -> float:
    """Fraction of content query terms attested (with light stemming) in `text`."""
    q_terms = _query_terms(query)
    if not q_terms:
        return 1.0
    s_terms = _tokens(text)
    hit = 0
    for q in q_terms:
        if any(_related({q}, w) for w in s_terms):
            hit += 1
    return hit / len(q_terms)


def grounding_verdict(query: str, extracted: ExtractiveAnswer) -> tuple[bool, str, float]:
    """Return (ok, reason, coverage). ok=False means abstain — do not serve or polish."""
    if extracted.is_empty:
        return False, "empty_span", 0.0
    cov = query_coverage(query, extracted.text)
    if extracted.support < MIN_SUPPORT:
        return False, f"low_support({extracted.support:.3f}<{MIN_SUPPORT})", cov
    if cov < MIN_COVERAGE:
        return False, f"low_query_coverage({cov:.3f}<{MIN_COVERAGE})", cov
    return True, "ok", cov


def extract_answer(
    query: str,
    documents: list[str],
    similarities: list[float] | None = None,
    top_docs: int = 3,
) -> ExtractiveAnswer:
    """Pick the best-supported sentence across the top retrieved passages."""
    t0 = time.perf_counter()
    docs = [d.strip() for d in (documents or []) if d and str(d).strip()]
    if not docs:
        return ExtractiveAnswer("", 0.0, "", 0, 0.0)

    docs = docs[:top_docs]
    sims = list(similarities or [])
    while len(sims) < len(docs):
        sims.append(0.0)

    q_terms = _query_terms(query)
    definition_query = bool(_DEF_QUERY.search(query or ""))
    best_text = ""
    best_score = -1.0
    best_src = docs[0]
    n_candidates = 0

    for rank, (doc, sim) in enumerate(zip(docs, sims)):
        example_passage = bool(_EXAMPLE_PASSAGE.search(doc))
        for sent in _split_sentences(doc):
            n_candidates += 1
            score = _score_sentence(
                sent,
                q_terms,
                rank,
                float(sim),
                example_passage=example_passage,
                definition_query=definition_query,
            )
            if score > best_score:
                best_score = score
                best_text = sent
                best_src = doc

    if not best_text:
        fallback = ""
        for doc in docs:
            for sent in _split_sentences(doc):
                if _BOILERPLATE.search(sent) or _EXAMPLE_PASSAGE.search(doc):
                    continue
                if len(sent) > len(fallback):
                    fallback, best_src = sent, doc
        best_text = fallback or docs[0][:400]
        best_score = 0.0

    return ExtractiveAnswer(
        text=best_text,
        support=round(float(max(best_score, 0.0)), 4),
        source_passage=best_src,
        n_candidates=n_candidates,
        took_ms=round((time.perf_counter() - t0) * 1000.0, 3),
    )
