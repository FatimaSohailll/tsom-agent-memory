## fc_sh32k_tsom_s1 FactConsolidation-sh 32k (tsom, qwen2.5:3b, 1 samples, 100 questions)

| Sample | n | Accuracy | F1 |
|---|---|---|---|
| sh_32k | 100 | 0.790 | 0.763 |
| **All** | 100 | **0.790** | 0.763 |

Stale answers: 20 of 64 questions whose fact changed (0.312).

Active contradictions per 100 facts at the last checkpoint: 0.00, against 23.04 with no consolidation (lower at 114 of 116 checkpoints).

| Cost | Value |
|---|---|
| Ingest LLM calls | 1107 |
| Ingest tokens (prompt / completion) | 1181128 / 195168 |
| Answer LLM calls | 100 |
| Answer tokens (prompt / completion) | 709433 / 388 |
| Fact operations | ADD 1749, REAFFIRM 4, REPLACE 538 |
| Temporal failures | 0 |
