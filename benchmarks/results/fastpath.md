
Fast path  n=80  unique_prompts=30
Window: transcript → extractive (generate=false). STT and LLM excluded.
Budget: 50ms   80/80 under budget   badge=PASS  p100=PASS

**P50 1.525ms · P70 2.054ms · P100 7.257ms**

| Stage | P50 | P70 | P95 | P100 |
|---|---:|---:|---:|---:|
| stt | 0.008 | 0.008 | 0.01 | 0.089 |
| retrieval | 0.373 | 0.425 | 2.634 | 2.954 |
| guardrail | 0.021 | 0.023 | 1.445 | 7.113 |
| extract | 1.02 | 1.152 | 2.019 | 3.695 |
| fast_path | 1.525 | 2.054 | 3.93 | 7.257 |
| generation | 0.0 | 0.0 | 0.0 | 0.0 |
| hallucination_check | 0.0 | 0.0 | 0.0 | 0.0 |
| hedge_check | 0.0 | 0.0 | 0.0 | 0.0 |
| total | 1.553 | 2.091 | 3.975 | 7.272 |
| **fast_path** | **1.525** | **2.054** | 3.93 | **7.257** |

Answer sources: {'extractive': 74, 'refusal': 6}

Paste the P50/P70/P100 row into README. STT and Groq stay outside this table.
