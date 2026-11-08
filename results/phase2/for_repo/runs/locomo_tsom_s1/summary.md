## locomo_tsom_s1 (tsom, qwen2.5:3b, 2 conversations)

| Category | n | F1 | BLEU-1 |
|---|---|---|---|
| Multi-hop | 48 | 24.13 | 13.07 |
| Temporal | 59 | 33.17 | 26.07 |
| Open-domain | 13 | 22.26 | 20.20 |
| Single-hop | 117 | 25.76 | 20.72 |
| Adversarial | 64 | 35.06 | 31.79 |
| Overall | 301 | 28.78 | 22.88 |

| Cost | Value |
|---|---|
| Memory writes | 878 |
| LLM calls per write | 2.76 |
| Seconds per write | 3.80 |
| Writes that triggered evolution | 75.1% |
| Prompt tokens per question | 1521 |

| Per memory write | A-Mem write | Temporal | Total |
|---|---|---|---|
| LLM calls | 2.76 | 1.15 | 3.90 |
| Seconds | 3.80 | 0.28 | 4.08 |

Fact operations: ADD 53, REAFFIRM 5, REPLACE 127. Temporal steps that failed and were skipped: 0.

Active contradictions per 100 facts at the last checkpoint: 1.32, against 19.33 with no consolidation (lower at 27 of 28 checkpoints).
