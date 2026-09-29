# Phase 4 Results — Bonn D/E Full Validation

## Scope and fixed protocol

This Phase 4 run used all 200 Bonn segments: 100 `D_F/Fxxx` interictal segments (label D) and 100 `E_S/Sxxx` ictal segments (label E). Each complete segment was one sample. This is **segment-level validation**. No reliable patient identity was available, so these results do not support cross-patient, cross-subject, LOSO, or patient-independent generalization claims.

The Phase 3 pipeline settings were frozen: `delay_dim=64`, `svd_rank=15`, `koopman_coordinate_count=15`, `coordinate_source=V`, `reference_mode=origin`, Chebyshev primary deviation, and finite-p secondary deviation with `p=20` and `abs(lambda)` weights. Sampling rate was 173.61 Hz. The signal was used as stored; no band decomposition, per-segment z-score, feature selection, or hyperparameter search was performed. Full settings and package versions are in [`config_snapshot.json`](outputs/bonn_full/config_snapshot.json).

## Numerical audit

| Check | Result |
|---|---:|
| Segments processed | 200/200 |
| D / E successful | 100 / 100 |
| Pipeline failures | 0 |
| Samples with NaN/Inf | 0 |
| SVD failures | 0/200 |
| Eigendecomposition failures | 0/200 |
| CV folds / OOF rows | 25 / 9,000 |

### Linearity error and spectral radius

Values are mean ± sample SD; median, IQR, and range follow.

| Diagnostic | Class | Mean ± SD | Median | IQR | Range |
|---|---|---:|---:|---:|---:|
| Linearity error | D | 0.15814 ± 0.02485 | 0.15074 | 0.02069 | 0.12692–0.23297 |
| Linearity error | E | 0.14372 ± 0.01798 | 0.14277 | 0.02115 | 0.09033–0.19638 |
| Spectral radius | D | 0.99332 ± 0.00296 | 0.99290 | 0.00298 | 0.98260–0.99961 |
| Spectral radius | E | 0.99421 ± 0.00232 | 0.99396 | 0.00338 | 0.98860–0.99896 |

The linearity-error distributions differ descriptively, with lower mean and median in E; this is a dynamical diagnostic, not a classifier-selected feature. See [`linearity_spectral_summary.csv`](outputs/bonn_full/linearity_spectral_summary.csv) and [`Figure_BonnFull_linearity_distribution.png`](outputs/bonn_full/Figure_BonnFull_linearity_distribution.png).

## Primary Koopman feature statistics

Two-sided Mann–Whitney U tests compare D and E. Effect is Cliff's delta `E minus D`; negative values indicate E tends lower. Holm correction was applied across these six pre-registered primary features.

| Feature | D mean ± SD | E mean ± SD | Cliff's δ | Holm adjusted p |
|---|---:|---:|---:|---:|
| Dmax | 0.07945 ± 0.01566 | 0.06683 ± 0.00852 | -0.5864 | 3.95e-12 |
| Dmin | 0.01059 ± 0.00290 | 0.01219 ± 0.00364 | +0.3366 | 1.18e-4 |
| Dmean | 0.03053 ± 0.00238 | 0.03107 ± 0.00102 | -0.0022 | 0.9795 |
| Dstd | 0.01035 ± 0.00271 | 0.00784 ± 0.00139 | -0.6922 | 1.67e-16 |
| Dpower | 0.0010518 ± 0.0000770 | 0.0010295 ± 0.0000555 | -0.3680 | 2.79e-5 |
| Dentropy | 3.5370 ± 0.1634 | 3.5787 ± 0.1262 | +0.1396 | 0.1767 |

Four of six features have Holm-adjusted `p < 0.05`: Dmax, Dmin, Dstd, and Dpower. Dmean and Dentropy do not. Full medians, IQRs, min/max, raw p-values, and adjusted p-values are in [`statistical_tests.csv`](outputs/bonn_full/statistical_tests.csv).

### Did the Phase 3 Dmax/Dstd separation persist?

The direction and strong group effects persisted, but the smoke test's completely non-overlapping ranges did **not** persist in the full cohort:

| Feature | Phase 3 smoke D range | Phase 3 smoke E range | Full D range | Full E range |
|---|---:|---:|---:|---:|
| Dmax | 0.07350–0.11565 | 0.05840–0.06637 | 0.05803–0.14854 | 0.05262–0.09025 |
| Dstd | 0.00936–0.01284 | 0.00670–0.00850 | 0.00724–0.01890 | 0.00592–0.01260 |

Full-cohort Cliff's deltas were -0.5864 for Dmax and -0.6922 for Dstd, with adjusted p-values `3.95e-12` and `1.67e-16`. Thus the smoke observation remains a strong distributional tendency, not a perfect segment-by-segment separation.

The six primary feature distributions and Dmax-vs-Dstd scatter are shown in [`Figure_BonnFull_Koopman_feature_distributions.png`](outputs/bonn_full/Figure_BonnFull_Koopman_feature_distributions.png) and [`Figure_BonnFull_Dmax_Dstd.png`](outputs/bonn_full/Figure_BonnFull_Dmax_Dstd.png).

## Raw baseline statistics

Bonn D/E has a pronounced group-level raw amplitude/power difference. Raw standard deviation was `65.62 ± 57.82` in D and `306.57 ± 147.96` in E (Cliff's δ `+0.9440`, Holm p `3.09e-30`). Mean-square power was `8,200.78 ± 22,192.89` vs `116,412.21 ± 100,817.15` (δ `+0.9452`, Holm p `3.09e-30`). Raw entropy also differed (δ `+0.3378`, Holm p `7.39e-5`); raw mean did not (δ `+0.0356`, Holm p `0.6645`). The raw distributions have overlap and the large standard deviations reflect skew/heterogeneity; the group differences should not be described as perfect separation.

The four baseline tests and full summaries are in [`raw_statistical_tests.csv`](outputs/bonn_full/raw_statistical_tests.csv); distributions are shown in [`Figure_BonnFull_raw_feature_distributions.png`](outputs/bonn_full/Figure_BonnFull_raw_feature_distributions.png).

## Scale-invariance control

For the pre-fixed Phase 3 examples F094 (D) and S026 (E), each signal was compared with `10*x(t)` using the same Hankel/SVD/V/Chebyshev pipeline. Across the six Koopman features, the largest relative difference was `1.04e-15` for F094 and `1.21e-14` for S026. Linearity error and spectral radius were also unchanged to floating-point precision. In contrast, raw mean and standard deviation scaled by 10 (`relative difference=9`), and raw power scaled by 100 (`relative difference=99`); histogram entropy did not change under positive scaling. This confirms scale invariance for these two fixed controls, not every possible input or a universal biological property.

Per-feature original/scaled values are recorded in [`scale_invariance_control.json`](outputs/bonn_full/scale_invariance_control.json).

## Segment-level classifier comparison

All feature sets and classifiers used the same 25 folds from RepeatedStratifiedKFold (5 folds × 5 repeats, seed 2021). LR and SVM scaling was fit within each training fold. Table entries are fold mean ± sample SD. Label E was the positive class. All six requested metrics are in [`cv_metrics.csv`](outputs/bonn_full/cv_metrics.csv) and the full summary is in [`classifier_summary.csv`](outputs/bonn_full/classifier_summary.csv).

| Feature set | Classifier | Balanced accuracy | ROC-AUC |
|---|---|---:|---:|
| KOOPMAN_PRIMARY | LogisticRegression | 0.790 ± 0.056 | 0.859 ± 0.053 |
| KOOPMAN_PRIMARY | SVM | 0.849 ± 0.045 | 0.895 ± 0.036 |
| KOOPMAN_PRIMARY | RUSBoost | 0.797 ± 0.053 | 0.864 ± 0.053 |
| RAW_BASELINE | LogisticRegression | 0.894 ± 0.045 | 0.976 ± 0.019 |
| RAW_BASELINE | SVM | 0.921 ± 0.044 | 0.978 ± 0.024 |
| RAW_BASELINE | RUSBoost | 0.939 ± 0.040 | 0.972 ± 0.025 |
| KOOPMAN_PLUS_RAW | LogisticRegression | 0.934 ± 0.041 | 0.985 ± 0.013 |
| KOOPMAN_PLUS_RAW | SVM | 0.949 ± 0.036 | 0.994 ± 0.011 |
| KOOPMAN_PLUS_RAW | RUSBoost | 0.983 ± 0.021 | 0.998 ± 0.005 |

Raw-only outperformed Koopman-only for all three classifiers. The combined feature set had higher mean balanced accuracy and ROC-AUC than raw-only for all three classifiers. On paired shared folds, combined balanced accuracy was higher than raw-only on 20/25 folds for LR, 15/25 for SVM, and 21/25 for RUSBoost; combined ROC-AUC was higher on 18/25, 17/25, and 19/25 folds respectively. Differences across repeated folds are descriptive because repeats reuse the same 200 segments; no fold-level significance test was used. See [`classifier_pairwise_comparison.csv`](outputs/bonn_full/classifier_pairwise_comparison.csv) and [`Figure_BonnFull_classifier_comparison.png`](outputs/bonn_full/Figure_BonnFull_classifier_comparison.png).

## Interpretation

**Phase 4 category: SUPPORTIVE, limited to this segment-level benchmark.** Koopman features show persistent D/E distribution differences after a scale-invariant representation check, and adding the fixed Koopman features improved the three classifiers' mean CV metrics over raw-only features with mostly positive paired-fold differences. The representation therefore carries information complementary to this simple raw summary baseline in these 200 segments.

This does **not** show that Koopman features outperform raw features alone: they do not in these runs. The raw EEG amplitude and power differences are already strong. Higher CV scores do not prove that Koopman captures a unique seizure mechanism, and the selected two-sample scale control does not prove universal invariance. No patient-level identity was established, so generalization beyond segment-level discrimination is unknown.

## Protocol deviations and scope boundary

No protocol deviation was recorded. All 200 segments were retained; the frozen Phase 3 Koopman definitions and pre-registered Phase 4 parameters were used; all nine model/feature-set combinations shared the same splits; no scaling was fit outside training folds; no feature selection, tuning, band decomposition, extra classifier, or test-result-driven changes were made. The original Phase 1–3 core math modules were not modified.

Phase 4 is complete. No full Bonn experiment beyond the specified D/E validation and no SEED experiment was started.
