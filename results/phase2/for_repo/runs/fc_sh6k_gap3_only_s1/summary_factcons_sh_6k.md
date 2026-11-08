## fc_sh6k_gap3_only_s1 FactConsolidation-sh 6k (tsom, qwen2.5:3b, 1 samples, 100 questions)

| Sample | n | Accuracy | F1 |
|---|---|---|---|
| sh_6k | 100 | 0.510 | 0.509 |
| **All** | 100 | **0.510** | 0.509 |

Stale answers: 47 of 74 questions whose fact changed (0.635).

Active contradictions per 100 facts at the last checkpoint: 24.60, against 24.60 with no consolidation (lower at 0 of 23 checkpoints).

| Cost | Value |
|---|---|
| Ingest LLM calls | 113 |
| Ingest tokens (prompt / completion) | 246718 / 32490 |
| Answer LLM calls | 100 |
| Answer tokens (prompt / completion) | 718844 / 401 |
| Fact operations | ADD 443, REAFFIRM 1 |
| Temporal failures | 0 |
