## fc_mh32k_tsom_s1 FactConsolidation-mh 32k (tsom, qwen2.5:3b, 1 samples, 100 questions)

| Sample | n | Accuracy | F1 |
|---|---|---|---|
| mh_32k | 100 | 0.060 | 0.072 |
| **All** | 100 | **0.060** | 0.072 |

Stale answers: 20 of 82 questions whose fact changed (0.244). Multi-hop estimate: only stale values of the last fact are counted.

Active contradictions per 100 facts at the last checkpoint: 0.00, against 23.04 with no consolidation (lower at 114 of 116 checkpoints).

| Cost | Value |
|---|---|
| Ingest LLM calls | 1107 |
| Ingest tokens (prompt / completion) | 1181128 / 195168 |
| Answer LLM calls | 100 |
| Answer tokens (prompt / completion) | 718524 / 455 |
| Fact operations | ADD 1749, REAFFIRM 4, REPLACE 538 |
| Temporal failures | 0 |
