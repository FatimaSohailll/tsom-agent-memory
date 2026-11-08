## fc_mh6k_tsom_s3 FactConsolidation-mh 6k (tsom, qwen2.5:3b, 1 samples, 100 questions)

| Sample | n | Accuracy | F1 |
|---|---|---|---|
| mh_6k | 100 | 0.150 | 0.162 |
| **All** | 100 | **0.150** | 0.162 |

Stale answers: 10 of 80 questions whose fact changed (0.125). Multi-hop estimate: only stale values of the last fact are counted.

Active contradictions per 100 facts at the last checkpoint: 0.00, against 24.60 with no consolidation (lower at 22 of 23 checkpoints).

| Cost | Value |
|---|---|
| Ingest LLM calls | 223 |
| Ingest tokens (prompt / completion) | 277729 / 35821 |
| Answer LLM calls | 100 |
| Answer tokens (prompt / completion) | 719841 / 388 |
| Fact operations | ADD 333, REAFFIRM 1, REPLACE 110 |
| Temporal failures | 0 |
