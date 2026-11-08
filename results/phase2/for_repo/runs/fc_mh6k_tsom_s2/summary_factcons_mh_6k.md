## fc_mh6k_tsom_s2 FactConsolidation-mh 6k (tsom, qwen2.5:3b, 1 samples, 100 questions)

| Sample | n | Accuracy | F1 |
|---|---|---|---|
| mh_6k | 100 | 0.190 | 0.202 |
| **All** | 100 | **0.190** | 0.202 |

Stale answers: 10 of 80 questions whose fact changed (0.125). Multi-hop estimate: only stale values of the last fact are counted.

Active contradictions per 100 facts at the last checkpoint: 0.00, against 25.39 with no consolidation (lower at 22 of 23 checkpoints).

| Cost | Value |
|---|---|
| Ingest LLM calls | 229 |
| Ingest tokens (prompt / completion) | 202588 / 34635 |
| Answer LLM calls | 100 |
| Answer tokens (prompt / completion) | 556959 / 405 |
| Fact operations | ADD 337, REAFFIRM 1, REPLACE 116 |
| Temporal failures | 0 |
