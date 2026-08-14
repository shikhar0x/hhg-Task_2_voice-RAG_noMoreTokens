import os, sys, re, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from harness.orchestrator import VoiceRAGOrchestrator
from guardrails.threshold_gate import stem_word

# Every false-positive id from the confusion matrix + the two MUST-NOT-REGRESS anchors.
FP_IDS = ["90836","55665","1090356","168868","290643","197590","265552","166290",
          "316415","187378","361332","110842","237853","144943","OOD_04",
          "233826","260880"]   # last two = eagle, cantaloupe (anchors)

# Build the id -> query map from test_queries.json + the OOD_04 query.
QUERY_MAP = {}
if os.path.exists("benchmarks/test_queries.json"):
    for it in json.load(open("benchmarks/test_queries.json")):
        q = it.get("eng_query") or it.get("indic_query")
        if q:
            QUERY_MAP[str(it.get("query_id"))] = q
QUERY_MAP["OOD_04"] = "Who won the FIFA World Cup in 1930 in Uruguay?"

QSTOP = {"what","when","where","which","who","whom","why","how","does","did","doing",
         "done","many","much","long","time","with","from","into","have","that","this",
         "your","their","there","here","will","would","could","should","been","were",
         "they","them","some","such","also","only","just","than","then","into","upon"}

def overlap_metric(answer, context):
    raw_a = set(re.findall(r'\b[a-zA-Z]{4,}\b', answer.lower())) - \
            {"this","that","from","with","have","were","they","their","about","would","which","there","these","where"}
    if not raw_a:
        return 0.0
    a = {stem_word(w) for w in raw_a}
    c = {stem_word(w) for w in set(re.findall(r'\b[a-zA-Z]{4,}\b', context.lower()))}
    return len(a & c) / len(a)

def query_coverage(query, context):
    salient = sorted({stem_word(w) for w in re.findall(r'\b[a-zA-Z]{4,}\b', query.lower()) if w not in QSTOP})
    if not salient:
        return [], [], 1.0
    cset = {stem_word(w) for w in set(re.findall(r'\b[a-zA-Z]{4,}\b', context.lower()))}
    covered = [t for t in salient if t in cset]
    missing = [t for t in salient if t not in cset]
    return salient, covered, len(covered)/len(salient)

def numbers_in_answer_not_in_ctx(answer, context):
    nums_a = set(re.findall(r'\b\d[\d,.\-]*\b', answer))
    nums_c = set(re.findall(r'\b\d[\d,.\-]*\b', context))
    return sorted(nums_a), sorted(nums_a - nums_c)

def main():
    orch = VoiceRAGOrchestrator()
    ret, gen, guard = orch.retrieval, orch.generation, orch.guardrail
    rows = []
    for qid in FP_IDS:
        q = QUERY_MAP.get(qid)
        if not q:
            print(f"\n### {qid}: query not found, skipping"); continue
        r = ret.run({"transcript": q})
        docs = r.data["documents"]; sims = r.data["similarities"]; top = r.data["top_similarity"]
        g = guard.execute({"query": q, "retrieval_result": r.data})
        if g.refused:
            rows.append((qid,q,top,sims,docs,None,g.refusal_reason,None,None,None,None))
            continue
        context = g.data["valid_context"]
        gg = gen.run({"transcript": q, "context": context})
        ans = gg.data.get("answer","")
        ov = overlap_metric(ans, context)
        sal, cov, covr = query_coverage(q, context)
        nsa, nmiss = numbers_in_answer_not_in_ctx(ans, context)
        rows.append((qid,q,top,sims,docs,ans,None,ov,(sal,cov,covr),(nsa,nmiss),None))

    for qid,q,top,sims,docs,ans,refusal,ov,qcov,nums,_ in rows:
        print("\n" + "="*100)
        kind = "ANCHOR (must ANSWER)" if qid in ("233826","260880") else "FALSE-POSITIVE (must REFUSE)"
        print(f"### {qid} | {kind}\n    Q: {q}")
        print(f"    top_similarity = {top:.4f}   | all sims = {[round(s,4) for s in sims]}")
        if refusal:
            print(f"    >>> GUARDRAIL REFUSED at retrieval: {refusal}")
            print(f"    (no generation happened)")
            continue
        for i,(d,s) in enumerate(zip(docs,sims)):
            print(f"    Passage {i+1} (sim {s:.4f}): {d}")
        print(f"    --- LLM ANSWER ---\n    {ans}")
        print(f"    current overlap(answer,ctx) = {ov:.3f}  (today's threshold = 0.20 -> {'PASS' if ov>=0.20 else 'REFUSE'})")
        sal,cov,covr = qcov
        print(f"    query salient terms: {sal}")
        print(f"      covered in ctx   : {cov}   (coverage = {covr:.2f})")
        nsa,nmiss = nums
        print(f"    numbers in answer  : {nsa}")
        print(f"      NOT in context   : {nmiss}   {'<-- FABRICATED NUMERIC SPECIFICS' if nmiss else ''}")
    print("\n" + "="*100)

if __name__ == "__main__":
    main()
