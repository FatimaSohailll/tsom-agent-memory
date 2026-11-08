## fc_sh6k_tsom_s3 FactConsolidation-sh 6k (tsom, qwen2.5:3b, 1 samples, 100 questions)

| Sample | n | Accuracy | F1 |
|---|---|---|---|
| sh_6k | 100 | 0.670 | 0.654 |
| **All** | 100 | **0.670** | 0.654 |

Stale answers: 31 of 74 questions whose fact changed (0.419).

Active contradictions per 100 facts at the last checkpoint: 0.00, against 24.60 with no consolidation (lower at 22 of 23 checkpoints).

| Cost | Value |
|---|---|
| Ingest LLM calls | 223 |
| Ingest tokens (prompt / completion) | 277729 / 35821 |
| Answer LLM calls | 100 |
| Answer tokens (prompt / completion) | 719691 / 417 |
| Fact operations | ADD 333, REAFFIRM 1, REPLACE 110 |
| Temporal failures | 0 |
