# Development log

Phase 2 requirement (Section 3.2): one entry per major iteration with the date, the failure or
limitation observed, the change made, and its measured effect. Only record numbers that exist
in `results/`; cite the run folder. Entries marked TODO need your own date and numbers.

| # | Date | Observed failure / limitation | Change made | Measured effect (run folder) |
|---|------|-------------------------------|-------------|------------------------------|
| 1 | 2026-10-17 | Phase 1 needed a no-LLM baseline next to A-Mem to show what memory building costs. | Added `tsom/rag.py` (raw turns, top-k retrieval, no LLM call on write). | LoCoMo conv 0, Qwen2.5-3B: A-Mem F1 22.11, 2.94 LLM calls/write, 4.10 s/write; RAG F1 12.26, 0 calls/write (`results/quick_test`, `results/quick_test_rag`). |
| 2 | TODO | A-Mem keeps the old and new value of a changed fact side by side (no conflict handling). | Gap 1 `FactLayer` with labels additive / redundant / corrective / contradictory. | LoCoMo conv 0: tsom_conflict F1 27.57 vs A-Mem 28.04; 3.97 vs 2.95 calls/write (`results/` TODO folder). Label accuracy on the first 6 scenarios was inconclusive (n too small). |
| 3 | TODO | Metric bug: the missed-correction denominator differed between modes (0/3 vs 1/4). | Same denominator for all modes; verdict says "inconclusive (n too small)" when n is small. | TODO |
| 4 | TODO | Add one entry per failure found in `scripts/error_dump.py` output: what failed, what changed, new numbers. | | |

Rules for new entries: write the date the change was made, link the commit, and re-run the same
evaluation before and after so the "measured effect" is a before/after pair.
