"""eval-loop adapter"""
from __future__ import annotations
import hashlib, re, time, importlib.util, sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np

_p = Path(__file__).resolve().parent / "retrieval" / "extractive.py"
_spec = importlib.util.spec_from_file_location("voice_rag_extractive", _p)
_ex = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _ex
_spec.loader.exec_module(_ex)
ABSTAIN_TEXT, extract_answer, grounding_verdict = _ex.ABSTAIN_TEXT, _ex.extract_answer, _ex.grounding_verdict

DIM = 2048
_TOKEN = re.compile(r"[\u0600-\u0D7F\w]+")
_STOP = {"a","an","the","and","or","of","to","in","on","for","is","are","was","were","be","been","being","it","this","that","with","as","by","at","from","what","which","who","how","when","where","why","does","did","do","define","definition","meaning","explain","क्या","कौन","कहाँ","कब","कैसे","क्यों","है","हैं","का","की","के"}

def _signed(key):
    d = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int.from_bytes(d[:4], "little") % DIM, (1.0 if d[4] & 1 else -1.0)

def _vec(text):
    v = np.zeros(DIM, np.float32)
    raw = (text or "").strip().lower()
    if not raw: return v
    pad = f"  {raw}  "
    for n in (3, 4):
        for i in range(max(0, len(pad)-n+1)):
            g = pad[i:i+n]
            if g.isspace(): continue
            i_, s = _signed(f"c{n}:{g}"); v[i_] += s
    for tok in _TOKEN.findall(raw):
        if len(tok) < 3: continue
        i_, s = _signed(f"w:{tok}"); v[i_] += s * (0.05 if tok in _STOP else 1.0)
    nrm = float(np.linalg.norm(v))
    if nrm: v /= nrm
    return v

def embed_one(text): return _vec(text)
def embed(texts):
    return np.zeros((0, DIM), np.float32) if not texts else np.vstack([_vec(t) for t in texts])
def get_model(): return "tfidf-hash-ngram-2048"

@dataclass
class GeneratedAnswer:
    text: str; grounded: bool; generation_ms: float; model: str

def generate_answer(query, results):
    t0 = time.perf_counter()
    docs, sims = [], []
    for h in results or []:
        t = getattr(h, "text", "") or ""
        if str(t).strip():
            docs.append(str(t)); sims.append(float(getattr(h, "score", 0.0) or 0.0))
    if not docs:
        return GeneratedAnswer(ABSTAIN_TEXT, False, (time.perf_counter()-t0)*1000, "extractive-empty")
    ex = extract_answer(query, docs, similarities=sims)
    ok, _, _ = grounding_verdict(query, ex)
    ms = (time.perf_counter()-t0)*1000
    if not ok:
        return GeneratedAnswer(ABSTAIN_TEXT, False, ms, "extractive-abstain")
    return GeneratedAnswer(ex.text, True, ms, "extractive")
