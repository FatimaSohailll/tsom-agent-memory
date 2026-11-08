# Results

## Extraction recall

| System | Recall |
|---|---|
| tsom | 45.5% [38.5, 52.5] (n=200) |

## Change scenarios

| System | Correct | Stale | Hedged | Past-period correct |
|---|---|---|---|---|
| rag | 90.0% [84.0, 96.0] (n=100) | 6.0% [2.0, 11.0] (n=100) | 0.0% [0.0, 0.0] (n=100) | 93.8% [85.4, 100.0] (n=48) |
| amem | 82.0% [74.0, 89.0] (n=100) | 5.0% [1.0, 10.0] (n=100) | 0.0% [0.0, 0.0] (n=100) | 97.9% [93.8, 100.0] (n=48) |
| tsom | 73.0% [65.0, 82.0] (n=100) | 4.0% [1.0, 8.0] (n=100) | 0.0% [0.0, 0.0] (n=100) | 91.7% [81.2, 97.9] (n=48) |
| tsom_history | 72.0% [64.0, 81.0] (n=100) | 4.0% [1.0, 8.0] (n=100) | 0.0% [0.0, 0.0] (n=100) | 91.7% [81.2, 97.9] (n=48) |
| gap1_only | 73.0% [65.0, 82.0] (n=100) | 4.0% [1.0, 8.0] (n=100) | 0.0% [0.0, 0.0] (n=100) | 93.8% [85.4, 100.0] (n=48) |
| gap3_only | 78.0% [70.0, 86.0] (n=100) | 4.0% [1.0, 8.0] (n=100) | 0.0% [0.0, 0.0] (n=100) | 95.8% [89.6, 100.0] (n=48) |

Stale, rag vs amem: inconclusive, the intervals overlap.
Stale, tsom vs amem: inconclusive, the intervals overlap.
Stale, tsom_history vs amem: inconclusive, the intervals overlap.
Stale, gap1_only vs amem: inconclusive, the intervals overlap.
Stale, gap3_only vs amem: inconclusive, the intervals overlap.

## FactConsolidation

### mh_32k

| System | Accuracy | Stale | Contradictions per 100 facts vs no consolidation |
|---|---|---|---|
| rag | 2.0% [0.0, 5.0] (n=100) | 28.0% [18.3, 37.8] (n=82) | n/a |
| amem | 2.0% [0.0, 5.0] (n=100) | 19.5% [11.0, 28.0] (n=82) | n/a |
| tsom | 6.0% [2.0, 11.0] (n=100) | 24.4% [15.9, 34.1] (n=82) | 0.0 vs 23.0 |
| tsom_history | missing | missing | n/a |
| gap1_only | missing | missing | n/a |
| gap3_only | missing | missing | n/a |

Stale on mh_32k, rag vs amem: inconclusive, the intervals overlap.
Stale on mh_32k, tsom vs amem: inconclusive, the intervals overlap.

### mh_6k

| System | Accuracy | Stale | Contradictions per 100 facts vs no consolidation |
|---|---|---|---|
| rag | 3.0 ± 0.0% [0.0, 7.0] (n=100) | 15.0 ± 0.0% [7.5, 22.5] (n=80) | n/a |
| amem | 5.3 ± 0.6% [1.7, 9.7] (n=100) | 17.9 ± 1.4% [10.0, 26.2] (n=80) | n/a |
| tsom | 16.3 ± 2.3% [10.0, 23.3] (n=100) | 12.5 ± 0.0% [5.8, 20.0] (n=80) | 0.0 ± 0.0 vs 24.9 ± 0.5 |
| tsom_history | missing | missing | n/a |
| gap1_only | 15.0% [9.0, 22.0] (n=100) | 11.2% [3.8, 20.0] (n=80) | 0.0 vs 24.6 |
| gap3_only | 5.0% [1.0, 9.0] (n=100) | 22.5% [13.8, 32.5] (n=80) | 24.6 vs 24.6 |

Stale on mh_6k, rag vs amem: inconclusive, the intervals overlap.
Stale on mh_6k, tsom vs amem: inconclusive, the intervals overlap.
Stale on mh_6k, gap1_only vs amem: inconclusive, the intervals overlap.
Stale on mh_6k, gap3_only vs amem: inconclusive, the intervals overlap.

### sh_32k

| System | Accuracy | Stale | Contradictions per 100 facts vs no consolidation |
|---|---|---|---|
| rag | 38.0% [28.0, 48.0] (n=100) | 53.1% [40.6, 65.6] (n=64) | n/a |
| amem | 42.0% [32.0, 51.0] (n=100) | 45.3% [32.8, 57.8] (n=64) | n/a |
| tsom | 79.0% [71.0, 87.0] (n=100) | 31.2% [20.3, 42.2] (n=64) | 0.0 vs 23.0 |
| tsom_history | missing | missing | n/a |
| gap1_only | missing | missing | n/a |
| gap3_only | missing | missing | n/a |

Stale on sh_32k, rag vs amem: inconclusive, the intervals overlap.
Stale on sh_32k, tsom vs amem: inconclusive, the intervals overlap.

### sh_6k

| System | Accuracy | Stale | Contradictions per 100 facts vs no consolidation |
|---|---|---|---|
| rag | 38.0 ± 0.0% [29.0, 48.0] (n=100) | 65.8 ± 0.8% [55.0, 76.6] (n=74) | n/a |
| amem | 40.0 ± 0.0% [30.7, 50.0] (n=100) | 65.8 ± 1.6% [55.9, 76.1] (n=74) | n/a |
| tsom | 67.0 ± 0.0% [58.0, 76.0] (n=100) | 41.9 ± 0.0% [31.1, 54.1] (n=74) | 0.0 ± 0.0 vs 24.6 ± 0.0 |
| tsom_history | missing | missing | n/a |
| gap1_only | 62.0% [53.0, 72.0] (n=100) | 45.9% [35.1, 58.1] (n=74) | 0.0 vs 24.6 |
| gap3_only | 51.0% [41.0, 60.0] (n=100) | 63.5% [52.7, 74.3] (n=74) | 24.6 vs 24.6 |

Stale on sh_6k, rag vs amem: inconclusive, the intervals overlap.
Stale on sh_6k, tsom vs amem: lower, and the intervals do not overlap.
Stale on sh_6k, gap1_only vs amem: inconclusive, the intervals overlap.
Stale on sh_6k, gap3_only vs amem: inconclusive, the intervals overlap.

Multi-hop stale rates are estimates: they count stale values of the last fact only.

## LoCoMo no-harm check

| System | Overall F1 (vs amem) | Temporal F1 (vs amem) |
|---|---|---|
| rag | 25.9 (-2.8) | 30.5 (-0.9) |
| amem | 28.7 | 31.4 |
| tsom | 28.8 (+0.1) | 33.2 (+1.8) |
| tsom_history | missing | missing |
| gap1_only | missing | missing |
| gap3_only | missing | missing |
