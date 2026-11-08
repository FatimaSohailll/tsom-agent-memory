| Gap | Method / component | Metric | Base paper result (A-Mem, reproduced) | Our result | Addressed? |
|---|---|---|---|---|---|
| Outdated and contradicted facts (Gap 1 + 3) | Fact layer: classifier + validity | Stale-answer rate, FactConsolidation SH 6K | 65.8% | 41.9% | fill in |
|  |  | Stale-answer rate, FactConsolidation MH 6K (estimate) | 17.9% | 12.5% | fill in |
|  |  | Accuracy, FactConsolidation SH 6K | 40.0% | 67.0% | fill in |
|  |  | Active contradictions per 100 facts, SH 6K | no facts stored | 0.00 vs 24.60 with nothing retired | fill in |
| Gap 1 component: update classifier | on vs off (gap3_only, always ADD) | Active contradictions per 100 facts, SH 6K | 24.60 vs 24.60 with nothing retired (classifier off) | 0.00 vs 24.60 with nothing retired | fill in |
| Gap 3 component: time-scoped retriever | on vs off (gap1_only) | Past-period questions correct, scenarios | 93.8% (retriever off) | 91.7% | fill in |
|  |  | Past-period questions correct, scenarios (vs A-Mem) | 97.9% | 91.7% | fill in |
| Write cost (Gap 2, measured only) | none (A-Mem's evolution unchanged) | LLM calls per write, FactConsolidation SH 6K | 2.91 | 9.70 | fill in |
