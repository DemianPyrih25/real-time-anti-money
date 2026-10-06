# Evaluation results

One alert = one flagged transaction. Alert budget = 0.5% of transactions (`configs/rules.yaml: alert_rate`, an analyst-capacity assumption). Model thresholds are chosen on val_late only and applied unchanged to test. Percentages are × 100; multi-seed models show mean ± std over seeds.

Brackets: 95% CI from a paired stratified cluster bootstrap (B = 2,000, seed 0) on the primary period: pattern positives resampled by attempt, other positives and negatives by sender account; (a) keeps its threshold, K is recomputed per replicate; multi-seed = mean over seeds per replicate.

## Headline: primary period

days 9-10: 862,792 transactions, 956 positives (0.111%)

| Model | Operating point | Alerts | Alerts/day | Alerted accounts/day | Precision % | Recall % | F1 % | PR-AUC % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Rules (SQL, 2 of 7 scenarios active) | own operating point | 4,110 | 2,055 | 1,757.5 | 1.6 [1.2, 2.0] | 6.9 [5.1, 9.1] | 2.6 [2.0, 3.3] | n/a |
| lgbm_tx (5 seeds) | (a) deployable threshold | 4,267.8 | 2,133.9 | 2,083.6 | 7.6 ± 0.2 [6.6, 8.6] | 33.7 ± 1.0 [28.7, 38.4] | 12.3 ± 0.3 [10.8, 13.9] | 7.4 ± 0.2 [5.8, 9.5] |
| lgbm_tx (5 seeds) | (b) top-K, K = rules' alerts | 4,110 | 2,055 | 2,007.2 | 7.8 ± 0.2 [6.7, 8.8] | 33.3 ± 0.7 [28.4, 38.1] | 12.6 ± 0.3 [10.9, 14.2] |  |
| lgbm_tx (5 seeds) | (c) rules ∪ model (a) | 8,296.4 | 4,148.2 | 3,753.7 | 4.2 ± 0.1 | 36.6 ± 0.9 [31.6, 41.5] | 7.6 ± 0.2 |  |
| lgbm_tx (5 seeds) | (c) model alone, same volume | 8,296.4 | 4,148.2 | 4,030.3 | 5.3 ± 0.0 | 46.4 ± 0.6 [40.2, 52.3] | 9.6 ± 0.1 |  |
| lgbm_graph (5 seeds) | (a) deployable threshold | 4,374 | 2,187 | 2,101.8 | 15.0 ± 0.1 [13.5, 16.4] | 68.4 ± 0.6 [60.0, 75.7] | 24.5 ± 0.2 [22.5, 26.6] | 57.4 ± 0.2 [50.0, 64.2] |
| lgbm_graph (5 seeds) | (b) top-K, K = rules' alerts | 4,110 | 2,055 | 1,972.5 | 15.8 ± 0.1 [13.9, 17.8] | 68.0 ± 0.5 [59.6, 75.1] | 25.7 ± 0.2 [23.0, 28.4] |  |
| lgbm_graph (5 seeds) | (c) rules ∪ model (a) | 8,217.8 | 4,108.9 | 3,731.7 | 8.0 ± 0.0 | 68.6 ± 0.5 [60.2, 75.9] | 14.3 ± 0.1 |  |
| lgbm_graph (5 seeds) | (c) model alone, same volume | 8,217.8 | 4,108.9 | 3,989.9 | 8.8 ± 0.1 | 75.4 ± 0.3 [66.3, 83.2] | 15.7 ± 0.1 |  |
| gnn_causal (5 seeds) | (a) deployable threshold | 5,524.8 | 2,762.4 | 2,647.1 | 13.2 ± 2.4 [12.1, 14.3] | 74.4 ± 1.5 [65.5, 82.3] | 22.4 ± 3.4 [20.7, 24.1] | 55.8 ± 2.6 [48.4, 62.5] |
| gnn_causal (5 seeds) | (b) top-K, K = rules' alerts | 4,110 | 2,055 | 1,953.3 | 16.7 ± 0.5 [14.8, 18.6] | 71.6 ± 2.1 [62.9, 79.3] | 27.0 ± 0.8 [24.5, 29.8] |  |
| gnn_causal (5 seeds) | (c) rules ∪ model (a) | 9,427.6 | 4,713.8 | 4,300.7 | 7.7 ± 0.8 | 75.0 ± 1.4 [65.9, 82.9] | 13.9 ± 1.3 |  |
| gnn_causal (5 seeds) | (c) model alone, same volume | 9,427.6 | 4,713.8 | 4,549.8 | 8.2 ± 0.9 | 79.7 ± 0.8 [70.1, 88.0] | 14.8 ± 1.6 |  |
| gnn_lookahead (3 seeds) | (a) deployable threshold | 5,126 | 2,563 | 2,439.7 | 14.5 ± 1.9 [13.3, 15.8] | 77.0 ± 1.1 [67.7, 84.9] | 24.4 ± 2.7 [22.5, 26.2] | 63.2 ± 0.3 [55.4, 70.1] |
| gnn_lookahead (3 seeds) | (b) top-K, K = rules' alerts | 4,110 | 2,055 | 1,948 | 17.4 ± 0.3 [15.5, 19.6] | 75.0 ± 1.2 [66.1, 82.9] | 28.3 ± 0.4 [25.6, 31.4] |  |
| gnn_lookahead (3 seeds) | (c) rules ∪ model (a) | 9,041 | 4,520.5 | 4,103.3 | 8.2 ± 0.6 | 77.2 ± 1.1 [67.8, 85.1] | 14.8 ± 1.0 |  |
| gnn_lookahead (3 seeds) | (c) model alone, same volume | 9,041 | 4,520.5 | 3,905.3 | 8.6 ± 0.7 | 80.7 ± 1.3 [71.1, 88.9] | 15.5 ± 1.1 |  |
| gnn_lookahead_d10 (3 seeds) | (a) deployable threshold | 5,101.3 | 2,550.7 | 2,429.8 | 14.5 ± 2.0 [13.3, 15.8] | 76.6 ± 1.0 [67.4, 84.5] | 24.4 ± 2.9 [22.5, 26.3] | 62.1 ± 0.7 [54.4, 68.9] |
| gnn_lookahead_d10 (3 seeds) | (b) top-K, K = rules' alerts | 4,110 | 2,055 | 1,949 | 17.4 ± 0.3 [15.5, 19.5] | 74.6 ± 1.3 [65.8, 82.5] | 28.2 ± 0.5 [25.5, 31.3] |  |
| gnn_lookahead_d10 (3 seeds) | (c) rules ∪ model (a) | 9,017 | 4,508.5 | 4,093.8 | 8.2 ± 0.7 | 76.9 ± 1.0 [67.6, 85.0] | 14.8 ± 1.1 |  |
| gnn_lookahead_d10 (3 seeds) | (c) model alone, same volume | 9,017 | 4,508.5 | 3,897.7 | 8.6 ± 0.7 | 80.6 ± 1.3 [71.1, 88.8] | 15.5 ± 1.2 |  |

- lgbm_tx: val_late alert rate at (a) = 0.466, 0.465, 0.466, 0.466, 0.466% per seed (rules: 0.466%; tied scores at the cut are left out).
- lgbm_graph: val_late alert rate at (a) = 0.466, 0.466, 0.466, 0.466, 0.466% per seed (rules: 0.466%; tied scores at the cut are left out).
- gnn_causal: val_late alert rate at (a) = 0.466, 0.466, 0.466, 0.466, 0.466% per seed (rules: 0.466%; tied scores at the cut are left out).
- gnn_lookahead: val_late alert rate at (a) = 0.466, 0.466, 0.466% per seed (rules: 0.466%; tied scores at the cut are left out).
- gnn_lookahead_d10: val_late alert rate at (a) = 0.466, 0.466, 0.466% per seed (rules: 0.466%; tied scores at the cut are left out).

### Paired differences (primary)

| Model | Recall (a) − rules, pp | Recall (b) − rules, pp | Recall union − model at same volume, pp |
| --- | --- | --- | --- |
| lgbm_tx (5 seeds) | 26.8 [22.0, 31.6] | 26.4 [21.6, 31.3] | -9.8 [-12.0, -7.6] |
| lgbm_graph (5 seeds) | 61.5 [53.4, 68.7] | 61.1 [53.0, 68.1] | -6.8 [-8.6, -5.3] |
| gnn_causal (5 seeds) | 67.5 [59.0, 75.2] | 64.7 [56.4, 72.1] | -4.7 [-5.7, -3.8] |
| gnn_lookahead (3 seeds) | 70.0 [61.4, 77.8] | 68.1 [59.8, 75.7] | -3.5 [-4.5, -2.8] |
| gnn_lookahead_d10 (3 seeds) | 69.7 [61.1, 77.4] | 67.7 [59.4, 75.4] | -3.7 [-4.8, -2.8] |

### Paired model differences (primary)

| Models | Recall (a), pp | Precision (a), pp | Recall (b), pp | F1 @ val_late-best threshold, pp | PR-AUC, pp | F1 @ 0.5 (argmax), pp |
| --- | --- | --- | --- | --- | --- | --- |
| lgbm_graph - lgbm_tx | 34.7 [29.7, 39.9] | 7.4 [6.5, 8.4] | 34.6 [29.6, 39.8] | 46.0 [41.3, 50.1] | 50.0 [43.5, 55.8] | 52.5 [45.9, 58.6] |
| gnn_causal - lgbm_tx | 40.7 [35.1, 46.3] | 5.7 [4.9, 6.5] | 38.3 [32.9, 43.6] | 43.1 [38.9, 47.0] | 48.4 [42.0, 54.2] | 44.7 [41.0, 48.1] |
| gnn_lookahead - lgbm_tx | 43.2 [37.5, 49.0] | 7.0 [6.2, 7.8] | 41.7 [36.2, 47.5] | 51.9 [47.1, 56.1] | 55.8 [48.9, 61.7] | 60.1 [55.4, 64.3] |
| gnn_lookahead_d10 - lgbm_tx | 42.9 [37.2, 48.6] | 7.0 [6.2, 7.8] | 41.3 [35.8, 47.3] | 50.9 [46.0, 55.3] | 54.7 [48.0, 60.6] | 59.7 [54.9, 63.9] |
| gnn_causal - lgbm_graph | 6.0 [4.0, 8.1] | -1.7 [-2.2, -1.2] | 3.6 [1.8, 5.6] | -2.9 [-4.7, -1.2] | -1.7 [-3.5, 0.1] | -7.8 [-11.7, -3.6] |
| gnn_lookahead - lgbm_graph | 8.5 [6.4, 10.9] | -0.4 [-1.0, 0.1] | 7.0 [5.1, 9.5] | 5.9 [3.9, 7.9] | 5.7 [4.2, 7.6] | 7.6 [4.1, 11.3] |
| gnn_lookahead_d10 - lgbm_graph | 8.2 [6.1, 10.6] | -0.4 [-1.0, 0.1] | 6.6 [4.7, 9.2] | 4.9 [2.9, 7.1] | 4.6 [3.0, 6.5] | 7.2 [3.8, 11.0] |
| gnn_lookahead - gnn_causal | 2.5 [1.3, 3.9] | 1.3 [1.0, 1.6] | 3.4 [2.1, 5.0] | 8.8 [7.0, 10.7] | 7.4 [5.6, 9.3] | 15.4 [13.4, 17.4] |
| gnn_lookahead_d10 - gnn_causal | 2.2 [1.0, 3.5] | 1.3 [1.0, 1.6] | 3.0 [1.8, 4.7] | 7.8 [5.9, 9.7] | 6.3 [4.6, 8.1] | 15.0 [13.0, 17.0] |
| gnn_lookahead_d10 - gnn_lookahead | -0.3 [-0.8, 0.1] | 0.0 [-0.1, 0.1] | -0.4 [-0.9, 0.2] | -1.0 [-1.9, -0.1] | -1.1 [-1.9, -0.4] | -0.4 [-1.1, 0.3] |

Same replicates for every model: the later model minus the earlier one.

## Literature-comparable metrics

**primary** (headline, days 9-10)

| Model | F1 @ val_late-best threshold | F1 @ 0.5 (argmax) | Precision @ 0.5 | Recall @ 0.5 | Alerts @ 0.5 | Precision @ thr | Recall @ thr | PR-AUC | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Rules (SQL, 2 of 7 scenarios active) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| lgbm_tx (5 seeds) | 14.5 ± 0.5 [12.3, 16.8] | 0.7 ± 1.0 [0.3, 1.2] | 33.0 ± 11.1 † | 0.4 ± 0.5 | 10.2 | 11.8 ± 1.5 | 19.3 ± 1.8 | 7.4 ± 0.2 [5.8, 9.5] | 92.3 ± 0.2 |
| lgbm_graph (5 seeds) | 60.5 ± 0.6 [54.9, 65.5] | 53.2 ± 0.3 [46.6, 59.3] | 95.4 ± 0.6 | 36.9 ± 0.2 | 369.8 | 76.8 ± 2.6 | 49.9 ± 0.7 | 57.4 ± 0.2 [50.0, 64.2] | 98.4 ± 0.0 |
| gnn_causal (5 seeds) | 57.6 ± 2.7 [52.5, 62.1] | 45.4 ± 5.8 [41.8, 48.7] | 37.2 ± 7.5 | 59.5 ± 0.9 | 1,593.2 | 67.6 ± 3.6 | 50.3 ± 2.7 | 55.8 ± 2.6 [48.4, 62.5] | 98.5 ± 0.2 |
| gnn_lookahead (3 seeds) | 66.4 ± 0.5 [60.8, 71.1] | 60.8 ± 1.0 [56.1, 65.0] | 63.2 ± 2.4 | 58.7 ± 1.4 | 889 | 84.1 ± 0.3 | 54.8 ± 0.8 | 63.2 ± 0.3 [55.4, 70.1] | 98.7 ± 0.1 |
| gnn_lookahead_d10 (3 seeds) | 65.4 ± 0.6 [59.6, 70.2] | 60.4 ± 0.7 [55.6, 64.6] | 63.7 ± 2.4 | 57.5 ± 1.0 | 864.7 | 85.0 ± 0.3 | 53.2 ± 0.7 | 62.1 ± 0.7 [54.4, 68.9] | 98.7 ± 0.1 |

**full** (days 9-18: the test period the published numbers use; no CI, the bootstrap covers the primary period only)

| Model | F1 @ val_late-best threshold | F1 @ 0.5 (argmax) | Precision @ 0.5 | Recall @ 0.5 | Alerts @ 0.5 | Precision @ thr | Recall @ thr | PR-AUC | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Rules (SQL, 2 of 7 scenarios active) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| lgbm_tx (5 seeds) | 22.6 ± 0.4 | 0.9 ± 1.2 | 49.0 ± 10.1 † | 0.5 ± 0.6 | 14.8 | 21.8 ± 2.1 | 24.0 ± 2.7 | 15.0 ± 0.1 | 94.9 ± 0.1 |
| lgbm_graph (5 seeds) | 74.0 ± 0.3 | 68.1 ± 0.3 | 96.9 ± 0.2 | 52.6 ± 0.3 | 874 | 86.3 ± 1.5 | 64.8 ± 0.8 | 74.2 ± 0.1 | 99.1 ± 0.0 |
| gnn_causal (5 seeds) | 73.2 ± 1.8 | 62.2 ± 5.5 | 53.6 ± 7.9 | 75.0 ± 0.6 | 2,298.4 | 79.7 ± 2.0 | 67.8 ± 2.5 | 74.1 ± 1.6 | 99.1 ± 0.1 |
| gnn_lookahead (3 seeds) | 78.7 ± 1.0 | 74.9 ± 1.0 | 76.1 ± 1.5 | 73.8 ± 1.8 | 1,563 | 89.3 ± 0.8 | 70.3 ± 2.0 | 78.3 ± 0.8 | 99.2 ± 0.1 |

Rules are binary, so PR-AUC and threshold F1 are n/a; their F1 at their own operating point is in the headline table. ROC-AUC is shown, not headlined. Compare published argmax F1 with the full-view F1 @ 0.5.

## Test view: tail

days 11-18: 1,108 transactions, 655 positives (59.116%)

| Model | Operating point | Alerts | Alerts/day | Alerted accounts/day | Precision % | Recall % | F1 % | PR-AUC % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Rules (SQL, 2 of 7 scenarios active) | own operating point | 43 | 5.4 | 5 | 97.7 | 6.4 | 12.0 | n/a |
| lgbm_tx (5 seeds) | (a) deployable threshold | 362.8 | 45.4 | 35.9 | 98.5 ± 0.1 | 54.5 ± 0.8 | 70.2 ± 0.7 | 97.1 ± 0.3 |
| lgbm_tx (5 seeds) | (b) top-K, K = rules' alerts | 43 | 5.4 | 4.5 | 95.3 ± 0.0 | 6.3 ± 0.0 | 11.7 ± 0.0 |  |
| lgbm_tx (5 seeds) | (c) rules ∪ model (a) | 385.6 | 48.2 | 38.2 | 98.5 ± 0.1 | 58.0 ± 0.8 | 73.0 ± 0.6 |  |
| lgbm_tx (5 seeds) | (c) model alone, same volume | 385.6 | 48.2 | 37.6 | 98.4 ± 0.0 | 58.0 ± 0.8 | 73.0 ± 0.6 |  |
| lgbm_graph (5 seeds) | (a) deployable threshold | 701.6 | 87.7 | 58.8 | 91.9 ± 0.4 | 98.4 ± 0.1 | 95.0 ± 0.2 | 97.8 ± 0.0 |
| lgbm_graph (5 seeds) | (b) top-K, K = rules' alerts | 43 | 5.4 | 3.3 | 97.7 ± 0.0 | 6.4 ± 0.0 | 12.0 ± 0.0 |  |
| lgbm_graph (5 seeds) | (c) rules ∪ model (a) | 701.6 | 87.7 | 58.8 | 91.9 ± 0.4 | 98.4 ± 0.1 | 95.0 ± 0.2 |  |
| lgbm_graph (5 seeds) | (c) model alone, same volume | 701.6 | 87.7 | 58.8 | 91.9 ± 0.4 | 98.4 ± 0.1 | 95.0 ± 0.2 |  |
| gnn_causal (5 seeds) | (a) deployable threshold | 735.4 | 91.9 | 59.3 | 88.4 ± 3.9 | 99.1 ± 0.2 | 93.4 ± 2.2 | 96.6 ± 0.2 |
| gnn_causal (5 seeds) | (b) top-K, K = rules' alerts | 43 | 5.4 | 3.6 | 98.6 ± 2.1 | 6.5 ± 0.1 | 12.1 ± 0.3 |  |
| gnn_causal (5 seeds) | (c) rules ∪ model (a) | 735.8 | 92.0 | 59.4 | 88.4 ± 3.9 | 99.2 ± 0.1 | 93.5 ± 2.2 |  |
| gnn_causal (5 seeds) | (c) model alone, same volume | 735.8 | 92.0 | 59.4 | 88.4 ± 3.9 | 99.1 ± 0.2 | 93.4 ± 2.2 |  |
| gnn_lookahead (3 seeds) | (a) deployable threshold | 704 | 88 | 58.8 | 91.6 ± 0.6 | 98.5 ± 1.6 | 94.9 ± 0.4 | 97.9 ± 0.3 |
| gnn_lookahead (3 seeds) | (b) top-K, K = rules' alerts | 43 | 5.4 | 3 | 100.0 ± 0.0 | 6.6 ± 0.0 | 12.3 ± 0.0 |  |
| gnn_lookahead (3 seeds) | (c) rules ∪ model (a) | 704.3 | 88.0 | 58.8 | 91.6 ± 0.6 | 98.5 ± 1.6 | 94.9 ± 0.5 |  |
| gnn_lookahead (3 seeds) | (c) model alone, same volume | 704.3 | 88.0 | 58.8 | 91.6 ± 0.6 | 98.5 ± 1.6 | 94.9 ± 0.5 |  |

## Test view: full

days 9-18: 863,900 transactions, 1,611 positives (0.186%)

| Model | Operating point | Alerts | Alerts/day | Alerted accounts/day | Precision % | Recall % | F1 % | PR-AUC % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Rules (SQL, 2 of 7 scenarios active) | own operating point | 4,153 | 415.3 | 355.5 | 2.6 | 6.7 | 3.7 | n/a |
| lgbm_tx (5 seeds) | (a) deployable threshold | 4,630.6 | 463.1 | 445.5 | 14.7 ± 0.2 | 42.2 ± 0.9 | 21.8 ± 0.3 | 15.0 ± 0.1 |
| lgbm_tx (5 seeds) | (b) top-K, K = rules' alerts | 4,153 | 415.3 | 400.0 | 15.5 ± 0.2 | 39.9 ± 0.6 | 22.3 ± 0.4 |  |
| lgbm_tx (5 seeds) | (c) rules ∪ model (a) | 8,682 | 868.2 | 781.3 | 8.4 ± 0.1 | 45.3 ± 0.8 | 14.2 ± 0.2 |  |
| lgbm_tx (5 seeds) | (c) model alone, same volume | 8,682 | 868.2 | 833.5 | 10.3 ± 0.1 | 55.6 ± 0.3 | 17.4 ± 0.1 |  |
| lgbm_graph (5 seeds) | (a) deployable threshold | 5,075.6 | 507.6 | 467.4 | 25.6 ± 0.2 | 80.6 ± 0.4 | 38.8 ± 0.2 | 74.2 ± 0.1 |
| lgbm_graph (5 seeds) | (b) top-K, K = rules' alerts | 4,153 | 415.3 | 377.0 | 30.7 ± 0.1 | 79.1 ± 0.3 | 44.2 ± 0.2 |  |
| lgbm_graph (5 seeds) | (c) rules ∪ model (a) | 8,919.4 | 891.9 | 793.4 | 14.6 ± 0.1 | 80.7 ± 0.3 | 24.7 ± 0.1 |  |
| lgbm_graph (5 seeds) | (c) model alone, same volume | 8,919.4 | 891.9 | 844.1 | 15.4 ± 0.1 | 85.2 ± 0.2 | 26.1 ± 0.2 |  |
| gnn_causal (5 seeds) | (a) deployable threshold | 6,260.2 | 626.0 | 576.9 | 22.2 ± 3.6 | 84.5 ± 1.0 | 35.1 ± 4.6 | 74.1 ± 1.6 |
| gnn_causal (5 seeds) | (b) top-K, K = rules' alerts | 4,153 | 415.3 | 371.0 | 31.6 ± 0.5 | 81.3 ± 1.4 | 45.5 ± 0.8 |  |
| gnn_causal (5 seeds) | (c) rules ∪ model (a) | 10,163.4 | 1,016.3 | 907.6 | 13.6 ± 1.4 | 84.8 ± 0.9 | 23.4 ± 2.0 |  |
| gnn_causal (5 seeds) | (c) model alone, same volume | 10,163.4 | 1,016.3 | 956.6 | 14.0 ± 1.5 | 87.8 ± 0.6 | 24.2 ± 2.3 |  |
| gnn_lookahead (3 seeds) | (a) deployable threshold | 5,830 | 583.0 | 534.9 | 23.9 ± 2.7 | 85.7 ± 1.2 | 37.3 ± 3.3 | 78.3 ± 0.8 |
| gnn_lookahead (3 seeds) | (b) top-K, K = rules' alerts | 4,153 | 415.3 | 372.4 | 32.5 ± 0.3 | 83.7 ± 0.7 | 46.8 ± 0.4 |  |
| gnn_lookahead (3 seeds) | (c) rules ∪ model (a) | 9,745.3 | 974.5 | 867.7 | 14.2 ± 0.9 | 85.8 ± 1.2 | 24.4 ± 1.3 |  |
| gnn_lookahead (3 seeds) | (c) model alone, same volume | 9,745.3 | 974.5 | 828.1 | 14.6 ± 1.0 | 88.0 ± 1.1 | 25.0 ± 1.4 |  |

## Sensitivity to the alert budget (primary period)

| Alert rate | Rule scenarios active | Rules val_late rate % | Rules alerts | Rules recall % | lgbm_tx (a) val_late rate % | lgbm_tx (a) alerts | lgbm_tx (a) recall % | lgbm_tx (b) recall % | lgbm_graph (a) val_late rate % | lgbm_graph (a) alerts | lgbm_graph (a) recall % | lgbm_graph (b) recall % | gnn_causal (a) val_late rate % | gnn_causal (a) alerts | gnn_causal (a) recall % | gnn_causal (b) recall % | gnn_lookahead (a) val_late rate % | gnn_lookahead (a) alerts | gnn_lookahead (a) recall % | gnn_lookahead (b) recall % | gnn_lookahead_d10 (a) val_late rate % | gnn_lookahead_d10 (a) alerts | gnn_lookahead_d10 (a) recall % | gnn_lookahead_d10 (b) recall % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5% | 2 of 7 | 0.466 | 4,110 | 6.9 | 0.466 | 4,267.8 | 33.7 ± 1.0 | 33.3 ± 0.7 | 0.466 | 4,374 | 68.4 ± 0.6 | 68.0 ± 0.5 | 0.466 | 5,524.8 | 74.4 ± 1.5 | 71.6 ± 2.1 | 0.466 | 5,126 | 77.0 ± 1.1 | 75.0 ± 1.2 | 0.466 | 5,101.3 | 76.6 ± 1.0 | 74.6 ± 1.3 |
| 0.1% | 2 of 7 | 0.086 | 783 | 5.1 | 0.085 | 725.8 | 12.7 ± 0.2 | 13.3 ± 0.2 | 0.086 | 722 | 52.0 ± 0.3 | 53.0 ± 0.5 | 0.086 | 887.4 | 53.0 ± 2.2 | 51.5 ± 2.3 | 0.086 | 861.7 | 58.4 ± 1.2 | 57.4 ± 0.8 | 0.086 | 836.7 | 57.3 ± 0.5 | 56.6 ± 0.3 |
| 1% | 3 of 7 | 0.931 | 8,401 | 9.3 | 0.930 | 8,671.6 | 47.1 ± 0.6 | 46.5 ± 0.6 | 0.931 | 8,748.2 | 76.1 ± 0.3 | 75.6 ± 0.2 | 0.931 | 13,484 | 81.9 ± 0.7 | 78.8 ± 1.8 | 0.931 | 12,692.3 | 82.4 ± 1.5 | 80.2 ± 1.6 | 0.931 | 12,659 | 82.4 ± 1.3 | 80.2 ± 1.6 |

(a) matches the rules' val_late alert rate; tied scores at the cut are left out, so its realised rate can be lower. (b) uses exactly the rules' test alert count.

## Rule scenarios at the headline budget (primary period, 0p005)

| Scenario | Alerts | True positives | Precision % | Recall % |
| --- | --- | --- | --- | --- |
| fan_in_velocity | off (not selected by tuning) | n/a | n/a | n/a |
| fan_out_velocity | off (not selected by tuning) | n/a | n/a | n/a |
| rapid_pass_through | 3,698 | 27 | 0.7 | 2.8 |
| round_trip | 418 | 41 | 9.8 | 4.3 |
| structuring | off (not selected by tuning) | n/a | n/a | n/a |
| round_amount_burst | off (not selected by tuning) | n/a | n/a | n/a |
| high_risk_format_burst | off (not selected by tuning) | n/a | n/a | n/a |

## Recall per typology at (a)

**primary** (days 9-10: 862,792 transactions, 956 positives (0.111%))

| Typology | Positives | Rules % | lgbm_tx % | lgbm_graph % | gnn_causal % | gnn_lookahead % | gnn_lookahead_d10 % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FAN-OUT | 65 | 4.6 | 59.7 ± 2.0 | 100.0 ± 0.0 | 96.3 ± 3.4 | 100.0 ± 0.0 | 100.0 ± 0.0 |
| FAN-IN | 57 | 5.3 | 54.7 ± 1.5 | 97.2 ± 1.0 | 97.9 ± 1.9 | 98.8 ± 1.0 | 97.7 ± 1.0 |
| CYCLE | 55 | 29.1 | 46.5 ± 4.4 | 94.5 ± 0.0 | 94.5 ± 2.6 | 98.2 ± 3.1 | 97.6 ± 2.8 |
| SCATTER-GATHER | 110 | 8.2 | 53.6 ± 1.8 | 98.9 ± 0.4 | 98.5 ± 0.8 | 100.0 ± 0.0 | 99.7 ± 0.5 |
| GATHER-SCATTER | 127 | 4.7 | 49.4 ± 3.2 | 97.8 ± 0.4 | 97.3 ± 0.9 | 97.9 ± 3.6 | 98.7 ± 0.5 |
| STACK | 84 | 7.1 | 61.2 ± 3.1 | 95.2 ± 1.2 | 91.7 ± 3.8 | 95.6 ± 0.7 | 94.0 ± 1.2 |
| BIPARTITE | 44 | 13.6 | 49.1 ± 2.0 | 93.2 ± 0.0 | 94.5 ± 3.0 | 93.2 ± 2.3 | 93.2 ± 2.3 |
| RANDOM | 43 | 32.6 | 47.9 ± 7.5 | 94.9 ± 1.0 | 93.0 ± 4.4 | 95.3 ± 2.3 | 93.8 ± 1.3 |
| OTHER | 371 | 0.8 | 3.0 ± 0.4 | 23.4 ± 1.0 | 40.5 ± 2.4 | 44.1 ± 2.7 | 43.9 ± 2.8 |
| ALL | 956 | 6.9 | 33.7 ± 1.0 | 68.4 ± 0.6 | 74.4 ± 1.5 | 77.0 ± 1.1 | 76.6 ± 1.0 |

**tail** (days 11-18: 1,108 transactions, 655 positives (59.116%))

| Typology | Positives | Rules % | lgbm_tx % | lgbm_graph % | gnn_causal % | gnn_lookahead % |
| --- | --- | --- | --- | --- | --- | --- |
| FAN-OUT | 66 | 0.0 | 67.6 ± 0.8 | 100.0 ± 0.0 | 99.4 ± 0.8 | 99.5 ± 0.9 |
| FAN-IN | 66 | 3.0 | 64.5 ± 2.0 | 100.0 ± 0.0 | 99.7 ± 0.7 | 99.5 ± 0.9 |
| CYCLE | 44 | 29.5 | 42.3 ± 2.6 | 100.0 ± 0.0 | 99.1 ± 2.0 | 99.2 ± 1.3 |
| SCATTER-GATHER | 129 | 3.1 | 52.1 ± 1.6 | 99.7 ± 0.4 | 99.8 ± 0.3 | 99.5 ± 0.4 |
| GATHER-SCATTER | 254 | 1.2 | 55.2 ± 1.7 | 98.8 ± 0.0 | 99.7 ± 0.4 | 97.9 ± 3.6 |
| STACK | 33 | 18.2 | 50.3 ± 3.5 | 86.7 ± 1.7 | 98.2 ± 2.7 | 94.9 ± 3.5 |
| BIPARTITE | 22 | 13.6 | 47.3 ± 2.5 | 95.5 ± 3.2 | 87.3 ± 5.0 | 93.9 ± 2.6 |
| RANDOM | 41 | 26.8 | 41.5 ± 3.0 | 96.1 ± 1.3 | 99.0 ± 1.3 | 100.0 ± 0.0 |
| OTHER | 0 | n/a | n/a | n/a | n/a | n/a |
| ALL | 655 | 6.4 | 54.5 ± 0.8 | 98.4 ± 0.1 | 99.1 ± 0.2 | 98.5 ± 1.6 |

**full** (days 9-18: 863,900 transactions, 1,611 positives (0.186%))

| Typology | Positives | Rules % | lgbm_tx % | lgbm_graph % | gnn_causal % | gnn_lookahead % |
| --- | --- | --- | --- | --- | --- | --- |
| FAN-OUT | 131 | 2.3 | 63.7 ± 1.2 | 100.0 ± 0.0 | 97.9 ± 1.5 | 99.7 ± 0.4 |
| FAN-IN | 123 | 4.1 | 60.0 ± 0.9 | 98.7 ± 0.4 | 98.9 ± 0.9 | 99.2 ± 0.8 |
| CYCLE | 99 | 29.3 | 44.6 ± 1.7 | 97.0 ± 0.0 | 96.6 ± 2.0 | 98.7 ± 1.5 |
| SCATTER-GATHER | 239 | 5.4 | 52.8 ± 1.5 | 99.3 ± 0.2 | 99.2 ± 0.4 | 99.7 ± 0.2 |
| GATHER-SCATTER | 381 | 2.4 | 53.3 ± 1.9 | 98.5 ± 0.1 | 98.9 ± 0.2 | 97.9 ± 3.6 |
| STACK | 117 | 10.3 | 58.1 ± 2.6 | 92.8 ± 0.8 | 93.5 ± 2.5 | 95.4 ± 1.3 |
| BIPARTITE | 66 | 13.6 | 48.5 ± 1.9 | 93.9 ± 1.1 | 92.1 ± 3.3 | 93.4 ± 2.3 |
| RANDOM | 84 | 29.8 | 44.8 ± 5.0 | 95.5 ± 1.0 | 96.0 ± 2.6 | 97.6 ± 1.2 |
| OTHER | 371 | 0.8 | 3.0 ± 0.4 | 23.4 ± 1.0 | 40.5 ± 2.4 | 44.1 ± 2.7 |
| ALL | 1,611 | 6.7 | 42.2 ± 0.9 | 80.6 ± 0.4 | 84.5 ± 1.0 | 85.7 ± 1.2 |

## Attempt-level detection at (a)

An attempt counts if it has at least one row in the view; it is detected if any of its rows is alerted. Minutes = median time from the attempt's first row in the view to its first alert (detected attempts only; for multi-seed models the mean over seeds of the per-seed median).

**primary**

| Typology | Attempts | Rules detected % | Rules minutes | lgbm_tx detected % | lgbm_tx minutes | lgbm_graph detected % | lgbm_graph minutes | gnn_causal detected % | gnn_causal minutes | gnn_lookahead detected % | gnn_lookahead minutes | gnn_lookahead_d10 detected % | gnn_lookahead_d10 minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FAN-OUT | 20 | 15.0 | 0 | 84.0 ± 2.2 | 0 | 100.0 ± 0.0 | 0 | 99.0 ± 2.2 | 0 | 100.0 ± 0.0 | 0 | 100.0 ± 0.0 | 0 |
| FAN-IN | 18 | 16.7 | 0 | 77.8 ± 0.0 | 0 | 100.0 ± 0.0 | 0 | 98.9 ± 2.5 | 0 | 100.0 ± 0.0 | 0 | 100.0 ± 0.0 | 0 |
| CYCLE | 22 | 59.1 | 84 | 46.4 ± 2.0 | 203.5 | 100.0 ± 0.0 | 0 | 98.2 ± 4.1 | 0 | 98.5 ± 2.6 | 0 | 98.5 ± 2.6 | 0 |
| SCATTER-GATHER | 22 | 36.4 | 0 | 89.1 ± 2.5 | 61.4 | 100.0 ± 0.0 | 0 | 100.0 ± 0.0 | 0 | 100.0 ± 0.0 | 0 | 100.0 ± 0.0 | 0 |
| GATHER-SCATTER | 37 | 10.8 | 0 | 78.4 ± 3.3 | 0 | 97.3 ± 0.0 | 0 | 99.5 ± 1.2 | 0 | 99.1 ± 1.6 | 0 | 100.0 ± 0.0 | 0 |
| STACK | 22 | 22.7 | 0 | 76.4 ± 5.0 | 0 | 100.0 ± 0.0 | 0 | 97.3 ± 4.1 | 0 | 98.5 ± 2.6 | 0 | 98.5 ± 2.6 | 0 |
| BIPARTITE | 16 | 37.5 | 0 | 82.5 ± 5.2 | 0 | 100.0 ± 0.0 | 0 | 100.0 ± 0.0 | 0 | 100.0 ± 0.0 | 0 | 100.0 ± 0.0 | 0 |
| RANDOM | 17 | 76.5 | 495 | 49.4 ± 7.9 | 0 | 94.1 ± 0.0 | 0 | 98.8 ± 2.6 | 0 | 94.1 ± 0.0 | 0 | 94.1 ± 0.0 | 0 |
| ALL | 174 | 31.6 | 0 | 73.6 ± 1.7 | 0 | 98.9 ± 0.0 | 0 | 99.0 ± 1.2 | 0 | 98.9 ± 0.6 | 0 | 99.0 ± 0.7 | 0 |

**tail**

| Typology | Attempts | Rules detected % | Rules minutes | lgbm_tx detected % | lgbm_tx minutes | lgbm_graph detected % | lgbm_graph minutes | gnn_causal detected % | gnn_causal minutes | gnn_lookahead detected % | gnn_lookahead minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FAN-OUT | 12 | 0.0 | n/a | 100.0 ± 0.0 | 0 | 100.0 ± 0.0 | 0 | 96.7 ± 4.6 | 0 | 97.2 ± 4.8 | 0 |
| FAN-IN | 13 | 7.7 | 0 | 96.9 ± 4.2 | 0 | 100.0 ± 0.0 | 0 | 100.0 ± 0.0 | 0 | 100.0 ± 0.0 | 0 |
| CYCLE | 11 | 63.6 | 173 | 67.3 ± 5.0 | 0 | 100.0 ± 0.0 | 0 | 100.0 ± 0.0 | 0 | 100.0 ± 0.0 | 0 |
| SCATTER-GATHER | 16 | 25.0 | 0 | 80.0 ± 5.2 | 76.4 | 100.0 ± 0.0 | 0 | 100.0 ± 0.0 | 0 | 100.0 ± 0.0 | 0 |
| GATHER-SCATTER | 27 | 7.4 | 1,477 | 91.9 ± 1.7 | 20.1 | 100.0 ± 0.0 | 0 | 100.0 ± 0.0 | 0 | 100.0 ± 0.0 | 0 |
| STACK | 12 | 33.3 | 819.5 | 88.3 ± 4.6 | 53.2 | 100.0 ± 0.0 | 0 | 100.0 ± 0.0 | 0 | 100.0 ± 0.0 | 0 |
| BIPARTITE | 9 | 33.3 | 0 | 75.6 ± 5.0 | 11.4 | 97.8 ± 5.0 | 0 | 100.0 ± 0.0 | 0 | 100.0 ± 0.0 | 0 |
| RANDOM | 9 | 66.7 | 558.5 | 53.3 ± 5.0 | 0 | 100.0 ± 0.0 | 0 | 100.0 ± 0.0 | 0 | 100.0 ± 0.0 | 0 |
| ALL | 109 | 24.8 | 156 | 84.2 ± 1.2 | 0 | 99.8 ± 0.4 | 0 | 99.6 ± 0.5 | 0 | 99.7 ± 0.5 | 0 |

**full**

| Typology | Attempts | Rules detected % | Rules minutes | lgbm_tx detected % | lgbm_tx minutes | lgbm_graph detected % | lgbm_graph minutes | gnn_causal detected % | gnn_causal minutes | gnn_lookahead detected % | gnn_lookahead minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FAN-OUT | 21 | 14.3 | 0 | 89.5 ± 2.1 | 0 | 100.0 ± 0.0 | 0 | 97.1 ± 2.6 | 0 | 98.4 ± 2.7 | 0 |
| FAN-IN | 19 | 21.1 | 0 | 94.7 ± 0.0 | 0 | 100.0 ± 0.0 | 0 | 98.9 ± 2.4 | 0 | 100.0 ± 0.0 | 0 |
| CYCLE | 22 | 68.2 | 307 | 57.3 ± 2.5 | 339.1 | 100.0 ± 0.0 | 0 | 98.2 ± 4.1 | 0 | 98.5 ± 2.6 | 0 |
| SCATTER-GATHER | 22 | 36.4 | 0 | 96.4 ± 3.8 | 139 | 100.0 ± 0.0 | 0 | 100.0 ± 0.0 | 0 | 100.0 ± 0.0 | 0 |
| GATHER-SCATTER | 37 | 13.5 | 0 | 88.1 ± 1.5 | 0 | 100.0 ± 0.0 | 0 | 100.0 ± 0.0 | 0 | 100.0 ± 0.0 | 0 |
| STACK | 25 | 36.0 | 0 | 79.2 ± 4.4 | 0 | 100.0 ± 0.0 | 0 | 98.4 ± 3.6 | 0 | 98.7 ± 2.3 | 0 |
| BIPARTITE | 20 | 45.0 | 0 | 75.0 ± 6.1 | 0 | 99.0 ± 2.2 | 0 | 100.0 ± 0.0 | 0 | 100.0 ± 0.0 | 0 |
| RANDOM | 17 | 94.1 | 650 | 56.5 ± 6.7 | 0 | 100.0 ± 0.0 | 0 | 100.0 ± 0.0 | 0 | 100.0 ± 0.0 | 0 |
| ALL | 183 | 37.7 | 0 | 80.7 ± 0.8 | 0 | 99.9 ± 0.2 | 0 | 99.1 ± 0.8 | 0 | 99.5 ± 0.5 | 0 |

## Memorisation check at (a)

Positives split by whether they touch an account that laundered in train.

**primary**

| Positives | Count | Rules recall % | lgbm_tx recall % | lgbm_graph recall % | gnn_causal recall % | gnn_lookahead recall % | gnn_lookahead_d10 recall % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| seen | 296 | 14.2 | 31.7 ± 2.6 | 58.2 ± 0.7 | 60.2 ± 0.5 | 60.6 ± 0.5 | 60.6 ± 0.5 |
| unseen | 660 | 3.6 | 34.6 ± 1.3 | 73.0 ± 0.7 | 80.8 ± 2.1 | 84.3 ± 1.4 | 83.8 ± 1.3 |

**tail**

| Positives | Count | Rules recall % | lgbm_tx recall % | lgbm_graph recall % | gnn_causal recall % | gnn_lookahead recall % |
| --- | --- | --- | --- | --- | --- | --- |
| seen | 113 | 17.7 | 70.1 ± 5.2 | 95.8 ± 0.7 | 99.8 ± 0.4 | 99.7 ± 0.5 |
| unseen | 542 | 4.1 | 51.3 ± 1.1 | 99.0 ± 0.1 | 99.0 ± 0.3 | 98.2 ± 1.8 |

**full**

| Positives | Count | Rules recall % | lgbm_tx recall % | lgbm_graph recall % | gnn_causal recall % | gnn_lookahead recall % |
| --- | --- | --- | --- | --- | --- | --- |
| seen | 409 | 15.2 | 42.3 ± 3.1 | 68.6 ± 0.5 | 71.1 ± 0.3 | 71.4 ± 0.4 |
| unseen | 1,202 | 3.8 | 42.1 ± 1.1 | 84.7 ± 0.4 | 89.0 ± 1.3 | 90.6 ± 1.5 |

## Which model wins (pre-registered rule)

Pre-registered in `configs/gnn.yaml: report` (hash `bce13f6a22ed`) before any `--final` run: d = gnn_causal − lgbm_graph on the primary period, paired bootstrap CI (95%). Per metric: the causal GNN wins it if the CI lies above 0, LightGBM-graph if below 0, else a tie.

| Metric | gnn_causal − lgbm_graph, pp | Outcome |
| --- | --- | --- |
| primary: recall (a) | 6.0 [4.0, 8.1] * | gnn |
| secondary: PR-AUC | -1.7 [-3.5, 0.1] | tie |

**Verdict: the causal GNN wins.**

## Look-ahead gap, step 1 (primary period, days 9-10)

The same model, sampler (`last`) and hyperparameters as the causal GNN, trained and scored with the look-ahead bound: a target's subgraph may hold edges up to the end of its split's period (train: the end of train; validation: the end of the validation days; test: the last edge of the data for `end`, the last edge of the primary period for `d10`), and the target's own edge is dropped. The causal bound admits strictly earlier minutes only. Gaps are look-ahead minus causal on the same paired bootstrap replicates; * marks a CI that excludes 0.

| Metric | gnn_causal (5 seeds) | Look-ahead, test bound end | Look-ahead, test bound d10 | Gap end − causal, pp | Gap d10 − causal, pp | Tail: end − d10, pp |
| --- | --- | --- | --- | --- | --- | --- |
| F1 @ val_late-best threshold | 57.6 ± 2.7 | 66.4 ± 0.5 | 65.4 ± 0.6 | 8.8 [7.0, 10.7] * | 7.8 [5.9, 9.7] * | 1.0 [0.1, 1.9] * |
| F1 @ 0.5 (argmax) | 45.4 ± 5.8 | 60.8 ± 1.0 | 60.4 ± 0.7 | 15.4 [13.4, 17.4] * | 15.0 [13.0, 17.0] * | 0.4 [-0.3, 1.1] |
| PR-AUC | 55.8 ± 2.6 | 63.2 ± 0.3 | 62.1 ± 0.7 | 7.4 [5.6, 9.3] * | 6.3 [4.6, 8.1] * | 1.1 [0.4, 1.9] * |

- val_early PR-AUC of the selected epochs (set summaries; ± = sample std over seeds): causal 61.1 ± 2.5 (seeds other than the HPO model seed, whose run repeats the selected trial: 61.2), look-ahead 66.8 ± 1.2.
- Future share (sampled edges ranked at or after the target's minute ÷ all sampled edges, look-ahead; test_d10 = the d10 test pass): train 65.7%, val_early 42.5%, val_late 18.8%, test 28.2%, test_d10 28.2%.
- With a far bound, `last` takes each busy node's latest edges of the split, whereas the published loader samples uniformly over the snapshot; the future share quantifies how much of a subgraph that is.
- Tail: end − d10 is what test edges after the primary period add to the look-ahead scores of the primary period.
- gnn_lookahead_d10 shares gnn_lookahead's validation scores; its tail rows carry causal-bound scores (so every score is finite), so it is shown on the primary period only.

## Faithful Multi-GNN reproduction (not in the model comparison)

Multi-GIN+EU under the published protocol as recalled from Multi-GNN's code (the paper does not state every setting; the recalled ones are listed in the faithful summary's `recalled`): non-temporal snapshot graphs that contain the target, uniform sampling, per-snapshot normalisation and a timestamp feature (the confined exemptions). One seed; never part of the comparison above.

**full** test view (days 9-18), argmax (score >= 0.5)

| Targets | F1 % | Precision % | Recall % | Rows |
| --- | --- | --- | --- | --- |
| sampled targets (published protocol) | 69.76 | 71.44 | 68.16 | 863,062 |
| all targets (ours: unsampled ones via the virtual target edge) | 69.76 | 71.44 | 68.16 | 863,900 |

Published Multi-GIN+EU on HI-Small: 64.79 ± 1.22 (Egressy et al., Table 2); pre-registered band [62.35, 67.23] (± 2 std). Verdict: **not reproduced**: F1 69.76 is 2.53 pp above the band [62.35, 67.23].

- Sampled share of test targets: 99.9%; by split (training summary): val_early 100.0%, val_late 100.0%, test 99.9%.
- Epochs run 100 of 100 (gate cap: none); best epoch 43 (0-based); batch size 8,192.
- Primary period (days 9-10, for information): F1 sampled targets 55.57, all targets 55.57.
- One seed: a correct pipeline misses a ± 2 std band about 5% of the time.

## As-of guard evidence (GNN)

Every sampled batch of every GNN run is checked by a code path independent of the sampler's: each sampled edge's rank must not exceed its target's bound, and a causal subgraph must not contain the target. Totals over every split the set scored:

- gnn_causal: 0 violations over 4,978,736,573 sampled edges; 0 target hits.
- gnn_lookahead: 0 violations over 6,217,851,653 sampled edges; 0 target hits (88,503,107 sampled copies of the target dropped).
- gnn_lookahead_d10: 0 violations over 6,217,389,535 sampled edges; 0 target hits (88,496,669 sampled copies of the target dropped).
- gnn_faithful (snapshot guard): 0 violations over 83,278,038,143 sampled edges (every sampled rank <= its snapshot's last rank; the target stays in its snapshot, as published).

## GNN training cost (estimate)

Per set: GPU-worker wall (gpu_seconds) x the exact shape price (GPU + cores + memory). A lower bound: drivers, container startups, failed attempts and the HPO search (`aml-hpo-gnn`) are not in it; reports/cost.md has the billed totals per Modal app (every protocol trains under `aml-train-gnn`).

- gnn_causal: ≈ $0.92 (0.74 h on NVIDIA L4, 4 cores, 32,768 MiB, at $1.2452/h).
- gnn_lookahead: ≈ $0.92 (0.74 h on NVIDIA L4, 4 cores, 32,768 MiB, at $1.2452/h).
- gnn_lookahead_d10: trained with gnn_lookahead (no cost of its own).
- gnn_faithful: ≈ $5.07 (3.53 h on NVIDIA L4, 8 cores, 32,768 MiB, at $1.4344/h).

## Validation-only M2 evidence

Chosen and measured before the test set is touched; no number below uses a test label (the feature-engine and rule-parity rows cover every split and use no labels).

### Feature gate (label-free, before any fit)

Kept 75 of 79 model inputs. Dropped engine features: 4 of 70 (5.7%; the gate stops for a decision above 25%). A feature is dropped if PSI(warm train days 4-6 vs val_early) > 0.25, if more than 1% of its val_early values fall outside the warm-train range, or if fewer than 100 train rows are non-zero. The warm-up PSI (days 1-3 vs 4-6) is a diagnostic only.

| Feature | Group | PSI | Warm-up PSI | Out of range % | Non-zero train rows | Gate |
| --- | --- | --- | --- | --- | --- | --- |
| log_amount_usd | TX | 0.0005 | 0.1053 | 0.00 | 3,248,921 | kept |
| payment_currency | TX | 0.0000 | 0.0001 | 0.00 | 3,161,454 | kept |
| receiving_currency | TX | 0.0000 | 0.0001 | 0.00 | 3,160,406 | kept |
| cross_currency | TX | 0.0000 | 0.0013 | 0.00 | 43,097 | kept |
| payment_format | TX | 0.0002 | 1.8580 | 0.00 | 2,885,337 | kept |
| self_loop | TX | 0.0000 | 0.6359 | 0.00 | 552,386 | kept |
| same_bank | TX | 0.0000 | 0.4712 | 0.00 | 612,743 | kept |
| round_amount | TX | 0.0000 | 0.0000 | 0.00 | 445 | kept |
| hour_of_day | TX | 0.0008 | 0.1304 | 0.00 | 2,746,107 | kept |
| u_out_cnt_1d | VEL | 0.0513 | 0.1447 | 0.00 | 2,513,525 | kept |
| u_out_cnt_3d | VEL | 0.1714 | 0.7132 | 0.00 | 2,714,259 | kept |
| u_out_uniq_1d | VEL | 0.0356 | 0.1417 | 0.00 | 2,513,525 | kept |
| u_out_uniq_3d | VEL | 0.0184 | 0.4556 | 0.00 | 2,714,259 | kept |
| u_in_cnt_1d | VEL | 0.0465 | 0.2752 | 0.00 | 1,937,225 | kept |
| u_in_cnt_3d | VEL | 0.2304 | 0.1904 | 0.00 | 2,408,979 | kept |
| u_in_uniq_1d | VEL | 0.0407 | 0.4314 | 0.00 | 1,937,225 | kept |
| u_in_uniq_3d | VEL | 0.2230 | 0.1707 | 0.00 | 2,408,979 | kept |
| v_in_cnt_1d | VEL | 0.0673 | 0.0255 | 0.00 | 2,452,009 | kept |
| v_in_cnt_3d | VEL | 0.3541 | 0.6109 | 0.00 | 2,789,336 | dropped (psi) |
| v_in_uniq_1d | VEL | 0.0474 | 0.2894 | 0.00 | 2,452,009 | kept |
| v_in_uniq_3d | VEL | 0.0946 | 0.6134 | 0.00 | 2,789,336 | kept |
| v_out_cnt_1d | VEL | 0.0207 | 0.2534 | 0.00 | 1,544,170 | kept |
| v_out_cnt_3d | VEL | 0.1592 | 0.5850 | 0.00 | 2,086,464 | kept |
| v_out_uniq_1d | VEL | 0.0179 | 0.1869 | 0.00 | 1,544,170 | kept |
| v_out_uniq_3d | VEL | 0.0122 | 0.2191 | 0.00 | 2,086,464 | kept |
| u_out_sum_1d | AMT | 0.0347 | 0.2556 | 0.00 | 2,512,248 | kept |
| u_out_sum_3d | AMT | 0.0179 | 0.5722 | 0.00 | 2,709,569 | kept |
| u_in_sum_1d | AMT | 0.0402 | 0.3502 | 0.00 | 1,929,589 | kept |
| u_in_sum_3d | AMT | 0.0621 | 0.2986 | 0.00 | 2,401,083 | kept |
| v_in_sum_1d | AMT | 0.0403 | 0.2209 | 0.00 | 2,449,906 | kept |
| v_in_sum_3d | AMT | 0.0526 | 0.6319 | 0.00 | 2,787,345 | kept |
| v_out_sum_1d | AMT | 0.0165 | 0.3598 | 0.00 | 1,544,160 | kept |
| v_out_sum_3d | AMT | 0.0234 | 0.3836 | 0.00 | 2,086,439 | kept |
| u_out_mean_1d | AMT | 0.0331 | 0.3136 | 0.00 | 2,513,525 | kept |
| u_out_mean_3d | AMT | 0.0424 | 0.6328 | 0.00 | 2,714,259 | kept |
| v_in_mean_1d | AMT | 0.0303 | 0.0558 | 0.00 | 2,452,009 | kept |
| v_in_mean_3d | AMT | 0.0346 | 0.4042 | 0.00 | 2,789,336 | kept |
| u_out_std_1d | AMT | 0.0374 | 0.4052 | 0.00 | 2,094,424 | kept |
| u_out_std_3d | AMT | 0.0615 | 1.0680 | 0.00 | 2,430,399 | kept |
| v_in_std_1d | AMT | 0.0541 | 0.3217 | 0.00 | 1,834,325 | kept |
| v_in_std_3d | AMT | 0.0655 | 1.2918 | 0.00 | 2,439,091 | kept |
| u_out_max_1d | AMT | 0.0292 | 0.2748 | 0.00 | 2,513,525 | kept |
| v_in_max_1d | AMT | 0.0357 | 0.2483 | 0.00 | 2,452,009 | kept |
| u_amt_dev_1d | AMT | 0.0173 | 0.2313 | 0.00 | 2,502,956 | kept |
| v_amt_dev_1d | AMT | 0.0279 | 0.2749 | 0.00 | 2,425,227 | kept |
| pair_cnt_1d | FLOW | 0.0424 | 0.1691 | 0.00 | 1,796,678 | kept |
| pair_cnt_3d | FLOW | 0.7276 | 1.1063 | 0.00 | 2,225,088 | dropped (psi) |
| u_inflow_12h | FLOW | 0.0125 | 0.0671 | 0.00 | 1,122,825 | kept |
| pt_ratio_12h | FLOW | 0.0116 | 0.0184 | 0.00 | 1,027,450 | kept |
| u_bal_3d | FLOW | 0.0138 | 1.0267 | 0.00 | 2,653,778 | kept |
| v_bal_3d | FLOW | 0.0328 | 1.5642 | 0.00 | 2,647,753 | kept |
| pair_is_new | PORT | 0.0030 | 0.9917 | 0.00 | 952,816 | kept |
| out_port | PORT | 0.0004 | 0.2653 | 0.00 | 2,262,316 | kept |
| in_port | PORT | 0.0007 | 0.1152 | 0.00 | 2,185,771 | kept |
| u_out_gap | PORT | 0.0242 | 1.2019 | 0.00 | 2,750,200 | kept |
| u_in_gap | PORT | 0.2200 | 2.2416 | 0.00 | 2,666,856 | kept |
| v_in_gap | PORT | 0.1289 | 1.5270 | 0.00 | 2,824,931 | kept |
| v_out_gap | PORT | 0.1377 | 2.5244 | 0.00 | 2,593,097 | kept |
| pair_gap | PORT | 0.2486 | 1.6150 | 0.00 | 2,296,105 | kept |
| rev_pair_gap | PORT | 0.0016 | 0.1311 | 0.00 | 186,013 | kept |
| cyc2_2d | CYC | 0.0000 | 0.0002 | 0.00 | 1,093 | kept |
| cyc3_2d | CYC | 0.0000 | 0.0000 | 0.00 | 30 | dropped (nonzero) |
| cyc4_2d | CYC | 0.0000 | 0.0000 | 0.00 | 8 | dropped (nonzero) |
| sg_mids_1d | SG | 0.0000 | 0.0000 | 0.00 | 235 | kept |
| sg_srcs_1d | SG | 0.0000 | 0.0000 | 0.00 | 235 | kept |
| gs_u_1d | SG | 0.0472 | 0.4213 | 0.00 | 1,769,887 | kept |
| gs_v_1d | SG | 0.0248 | 0.2879 | 0.00 | 1,381,168 | kept |
| in_band | RULE | 0.0000 | 0.0003 | 0.00 | 30,655 | kept |
| u_out_inband_1d | RULE | 0.0970 | 0.0566 | 0.00 | 370,098 | kept |
| u_out_round_1d | RULE | 0.0431 | 0.0338 | 0.00 | 115,590 | kept |
| u_out_newcp_1d | RULE | 0.0007 | 3.1282 | 0.00 | 1,396,839 | kept |
| u_out_fmt_ach_1d | RULE | 0.0173 | 0.0182 | 0.00 | 864,697 | kept |
| u_out_fmt_bitcoin_1d | RULE | 0.0008 | 0.0013 | 0.04 | 72,922 | kept |
| u_out_fmt_cash_1d | RULE | 0.0314 | 0.0338 | 0.00 | 1,060,190 | kept |
| u_out_fmt_cheque_1d | RULE | 0.0463 | 0.1951 | 0.00 | 1,998,829 | kept |
| u_out_fmt_credit_card_1d | RULE | 0.0326 | 0.1462 | 0.00 | 1,774,333 | kept |
| u_out_fmt_reinvestment_1d | RULE | 0.0000 | 2.3432 | 0.00 | 587,795 | kept |
| u_out_fmt_wire_1d | RULE | 0.0083 | 0.0066 | 0.00 | 443,625 | kept |
| v_in_same_fmt_1d | RULE | 0.0702 | 0.0377 | 0.00 | 1,423,086 | kept |

### Group ablation on val_early (tuned parameters fixed)

Average precision on val_early over seeds 0, 1, 2, mean ± std (percent). Δ = mean AP(variant) − mean AP(full), in percentage points. Pre-registered rule: drop the one group with the largest Δ only if Δ > 2 σ, where σ = 0.19 pp is the pooled seed std over all variants (bar = 0.39 pp); no_gate and nofmt are reported only.

| Variant | Inputs | val_early AP % | Δ pp | Decision |
| --- | --- | --- | --- | --- |
| full | 75 | 60.96 ± 0.07 | +0.00 | champion |
| -VEL | 60 | 59.96 ± 0.14 | -1.00 | not chosen |
| -AMT | 55 | 61.09 ± 0.08 | +0.12 | not chosen |
| -FLOW | 70 | 60.73 ± 0.08 | -0.23 | not chosen |
| -PORT | 66 | 54.12 ± 0.33 | -6.84 | not chosen |
| -CYC | 74 | 60.87 ± 0.26 | -0.10 | not chosen |
| -SG | 71 | 61.21 ± 0.22 | +0.24 | not chosen |
| -RULE | 63 | 60.42 ± 0.22 | -0.54 | not chosen |
| no_gate | 79 | 61.13 ± 0.08 | +0.16 | reported only |
| nofmt | 66 | 56.91 ± 0.24 | -4.06 | reported only |

Decision: the champion is full (largest Δ: -SG, +0.24 pp, under the 0.39 pp bar).

Note: full at the selection seed is the winning Optuna trial (the search maximum), so the deltas lean against dropping a group.

### TreeSHAP of the champion (seed 0, val_early)

Mean |contribution| in raw score (log-odds) units on 100,497 rows: all 497 positives and 100,000 uniformly sampled negatives. A group's value is the sum over its features.

| Group | Inputs | Mean abs (sample) | Mean abs (positives) |
| --- | --- | --- | --- |
| TX | 9 | 0.8999 | 4.4638 |
| PORT | 9 | 0.6686 | 3.0458 |
| VEL | 15 | 0.5567 | 0.7522 |
| AMT | 20 | 0.4771 | 0.9327 |
| RULE | 12 | 0.2723 | 0.6711 |
| FLOW | 5 | 0.2514 | 0.4798 |
| SG | 4 | 0.0029 | 0.0057 |
| CYC | 1 | 0.0014 | 0.1090 |

Top 20 features:

| Rank | Feature | Group | Mean abs (sample) | Mean abs (positives) |
| --- | --- | --- | --- | --- |
| 1 | payment_format | TX | 0.5342 | 3.1747 |
| 2 | u_out_uniq_3d | VEL | 0.2250 | 0.2066 |
| 3 | pair_is_new | PORT | 0.2085 | 0.9508 |
| 4 | pair_cnt_1d | FLOW | 0.1654 | 0.3107 |
| 5 | log_amount_usd | TX | 0.1391 | 0.7764 |
| 6 | self_loop | TX | 0.1163 | 0.2148 |
| 7 | u_out_gap | PORT | 0.1005 | 0.3896 |
| 8 | in_port | PORT | 0.0913 | 0.5749 |
| 9 | u_out_uniq_1d | VEL | 0.0870 | 0.0842 |
| 10 | pair_gap | PORT | 0.0831 | 0.2800 |
| 11 | u_out_fmt_credit_card_1d | RULE | 0.0723 | 0.0854 |
| 12 | out_port | PORT | 0.0701 | 0.1612 |
| 13 | u_out_fmt_cheque_1d | RULE | 0.0559 | 0.1436 |
| 14 | u_out_cnt_3d | VEL | 0.0543 | 0.0604 |
| 15 | u_out_inband_1d | RULE | 0.0493 | 0.0365 |
| 16 | v_out_gap | PORT | 0.0460 | 0.0816 |
| 17 | u_out_max_1d | AMT | 0.0438 | 0.0694 |
| 18 | v_in_gap | PORT | 0.0403 | 0.4304 |
| 19 | u_out_cnt_1d | VEL | 0.0401 | 0.0345 |
| 20 | u_out_sum_3d | AMT | 0.0373 | 0.0473 |

### Feature engine (build_features summary.json)

| Item | Value |
| --- | --- |
| engine_version | 1 |
| spec_hash | 1d005132db35b6fe |
| n_accounts | 515,088 |
| hub_cap | 115 |
| n_hubs | 493 |
| bench_gate | pass |
| cyc_trunc_share | 0 |
| flush_max_ms | 124.488 |
| flush_p99_ms | 14.5424 |
| max_restored_mb | 362.306 |
| memory_within_target | True |
| projected_memory_mb | 432.639 |
| replay_min | 7.64904 |
| restart_check_pass | True |
| rows | 5,078,345 |
| rule_trunc_share | 0 |
| sg_trunc_share | 0 |
| us_per_event_engine | 57.4251 |
| us_per_event_io | 9.52598 |

### Rule parity

Rule parity (engine severities vs the M1 SQL): 0 mismatches over 5,078,345 rows (0 rows with rule_trunc = 1 are excluded).

| Item | Value |
| --- | --- |
| ok | True |
| rows | 5,078,345 |
| mismatches_total | 0 |
| mismatches_validation | 0 |
| rule_trunc.rows | 0 |
| rule_trunc.share | 0 |
| rule_trunc.round_trip_lower_than_sql | 0 |
| inflow_c_max | 1,662,082,714,534 |
| hubs.equal | True |
| hubs.hub_cap_engine | 115 |
| hubs.n_hubs_engine | 493 |
| m1_regression.compared | True |
| m1_regression.thresholds_equal | True |
| m1_regression.flags_equal | True |
| m1_regression.severities_equal | True |
| flag_agreement.0p001.all | 1 |
| flag_agreement.0p001.val_early | 1 |
| flag_agreement.0p001.val_late | 1 |
| flag_agreement.0p005.all | 1 |
| flag_agreement.0p005.val_early | 1 |
| flag_agreement.0p005.val_late | 1 |
| flag_agreement.0p01.all | 1 |
| flag_agreement.0p01.val_early | 1 |
| flag_agreement.0p01.val_late | 1 |

## Literature reference (published numbers, not directly comparable)

| Method (published) | HI-Small minority-class F1 % |
| --- | --- |
| GIN | 28.7 |
| PNA | 56.8 |
| Multi-GIN+EU | 64.8 |
| Multi-PNA+EU | 68.2 |
| LightGBM+GFP | 62.9 |
| XGBoost+GFP | 63.2 |
| LightGBM, raw transaction features | 21.3 |

- Published figures are on the whole post-validation test period, which is our full view (days 9-18, tail included): compare them with our full-view F1 @ 0.5 row, not the primary one.
- Minority-class F1 at the argmax threshold (0.5), not at a threshold chosen on validation.
- GNN rows: non-causal. The published test graph holds all edges and the loaders have no time constraint; Multi-GNN also drops unsampled target edges from its HI-Small F1.
- The GFP rows probably use a different split.
- Sources: Altman et al. (arXiv 2306.16424), Egressy et al. (arXiv 2306.11586), Blanuša et al. (arXiv 2402.08593); see PLAN.md §12.

## Caveats

- IBM AML HI-Small is synthetic: labels are perfect and instantly available, there is no KYC or geography, and there may be shortcuts such as payment format (ACH carries most positives).
- "Causal" means features and sampling (as-of rule: only strictly earlier minutes), not label availability.
- The tail (days 11-18) has only 1,108 transactions, 655 of them laundering (59% of tail rows; 100% of these positives belong to pattern attempts), so full-test numbers are dominated by it. The primary period is the headline.
- † Mean over fewer seeds than the model has: on the other seeds the metric is undefined (no alerts for precision, no detected attempt for minutes). Per-seed values are in results.json.
