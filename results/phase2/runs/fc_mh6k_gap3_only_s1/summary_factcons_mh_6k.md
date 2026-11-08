## fc_mh6k_gap3_only_s1 FactConsolidation-mh 6k (tsom, qwen2.5:3b, 1 samples, 100 questions)

| Sample | n | Accuracy | F1 |
|---|---|---|---|
| mh_6k | 100 | 0.050 | 0.062 |
| **All** | 100 | **0.050** | 0.062 |

Stale answers: 18 of 80 questions whose fact changed (0.225). Multi-hop estimate: only stale values of the last fact are counted.

Active contradictions per 100 facts at the last checkpoint: 24.60, against 24.60 with no consolidation (lower at 0 of 23 checkpoints).

| Cost | Value |
|---|---|
| Ingest LLM calls | 113 |
| Ingest tokens (prompt / completion) | 246718 / 32490 |
| Answer LLM calls | 100 |
| Answer tokens (prompt / completion) | 719130 / 376 |
| Fact operations | ADD 443, REAFFIRM 1 |
| Temporal failures | 0 |
