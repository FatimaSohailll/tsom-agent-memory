| Run | Contradictions /100 facts (SH 6K) | Accuracy, FC SH 6K | Stale, FC SH 6K | Accuracy, FC MH 6K | Stale, FC MH 6K | Current correct, change+cancel | Stale, change+cancel | Added value kept | Past-period correct | Past question got the current value |
|---|---|---|---|---|---|---|---|---|---|---|
| amem | no facts | 40.0 [30, 50] (n=100) | 64.9 [54, 76] (n=74) | 5.0 [1, 9] (n=100) | 18.8 [10, 28] (n=80) | 88.0 [78, 96] (n=50) | 10.0 [2, 18] (n=50) | 52.0 [32, 72] (n=25) | 97.9 [94, 100] (n=48) | 0.0 [0, 0] (n=48) |
| gap3_only | 24.60 | 51.0 [41, 60] (n=100) | 63.5 [53, 74] (n=74) | 5.0 [1, 9] (n=100) | 22.5 [14, 32] (n=80) | 78.0 [66, 90] (n=50) | 8.0 [2, 16] (n=50) | 68.0 [48, 84] (n=25) | 95.8 [90, 100] (n=48) | 0.0 [0, 0] (n=48) |
| gap1_only | 0.00 | 62.0 [53, 72] (n=100) | 45.9 [35, 58] (n=74) | 15.0 [9, 22] (n=100) | 11.2 [4, 20] (n=80) | 80.0 [68, 90] (n=50) | 8.0 [2, 16] (n=50) | 40.0 [20, 60] (n=25) | 93.8 [85, 100] (n=48) | 2.1 [0, 8] (n=48) |
| tsom | 0.00 | 67.0 [58, 76] (n=100) | 41.9 [31, 54] (n=74) | 15.0 [9, 22] (n=100) | 12.5 [5, 21] (n=80) | 80.0 [68, 90] (n=50) | 8.0 [2, 16] (n=50) | 40.0 [20, 60] (n=25) | 91.7 [81, 98] (n=48) | 2.1 [0, 8] (n=48) |

Effect of each gap (full tsom minus the run with that gap off):

- Accuracy, FC SH 6K: Gap 1 +16.0 points, Gap 3 +5.0 points
- Stale, FC SH 6K: Gap 1 -21.6 points, Gap 3 -4.1 points
- Accuracy, FC MH 6K: Gap 1 +10.0 points, Gap 3 +0.0 points
- Stale, FC MH 6K: Gap 1 -10.0 points, Gap 3 +1.2 points
- Current correct, change+cancel: Gap 1 +2.0 points, Gap 3 +0.0 points
- Stale, change+cancel: Gap 1 +0.0 points, Gap 3 +0.0 points
- Added value kept: Gap 1 -28.0 points, Gap 3 +0.0 points
- Past-period correct: Gap 1 -4.2 points, Gap 3 -2.1 points
- Past question got the current value: Gap 1 +2.1 points, Gap 3 +0.0 points

Gap 3 cannot act without Gap 1 when the text states no dates: with the classifier off nothing is
replaced, so `gap3_only` can only filter by stated dates. Report that dependency with these numbers.
