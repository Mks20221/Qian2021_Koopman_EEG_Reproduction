# Current status

## Phase 0 — documentation and configuration

**Status: COMPLETE.** The independent project, corrected paper notes, protocol, and configuration were created. The Bonn folder mapping was user-confirmed before data audit.

## Phase 1 — core mathematical modules and unit tests

**Status: COMPLETE.** The explicit Hankel/SVD, intrinsic-coordinate candidates, least-squares DMD, eigenvalue diagnostics, linear reconstruction, and deviation candidates were implemented. The Phase 1 unit-test run recorded 36 passed, 0 failed. The Phase 1 core mathematical modules were not modified in Phases 2 or 3.

## Phase 2 — Lorenz reproduction and calibration

**Status: RUN COMPLETE; no numerical failure.** Phase 2 is a paper-reproduction/calibration stage. An unexpected result is recorded as `DEBUG` for investigation, not as a stop gate.

- Continuous L1 → L2 → L1 simulation used L1 `(sigma,rho,beta)=(10,28,8/3)` and L2 `(10,28,4/3)`, sampled at 100 Hz. The 20 s burn-in, `[1,1,1]` initial condition, 50 s per regime, and RK45 tolerances are explicit implementation choices.
- On the first post-burn-in L1 x-observation, fixed rank 15 gave relative one-step linearity errors `0.250362`, `0.136400`, `0.097263`, and `0.060869` for delay dimensions `16,32,64,128`. This is qualitatively consistent with the paper's stated embedding-dimension trend.
- On the full transition, delay 64 / rank 15 / V gave one-step relative error `0.0957534`, RMSE `0.00078348`, and spectral radius `0.996871`. All 149 1 s free-run windows remained finite; their mean relative error was `0.83816`.
- The two fixed D candidates distinguished the altered L2 interval under the origin-reference candidate; this is not asserted to be the paper's unique numerical Eq. (8) implementation. The `V`/`SigmaV`, finite-p weight mapping, and exact reference remain unresolved implementation details.
- Lorenz CSV/JSON metrics and figures are saved under `outputs/lorenz/`.

## Phase 3 — Bonn D/E smoke test

**Status: BONN SMOKE TEST COMPLETED — PASS.** This is a 10-segment numerical smoke result only. It is not a statistical or generalization claim.

- Audited all discovered D/E source files: D=`100/100`, E=`100/100`; every segment had `4097` samples. NaN count and Inf count were both zero. Pooled finite amplitude ranges/statistics: D min/max `-1147/2047`, mean `-6.203`, std `90.345`; E min/max `-1885/2047`, mean `-4.749`, std `341.159`.
- Seed 2021 selected D: `F094, F074, F073, F067, F049`; E: `S026, S004, S061, S057, S027`.
- All selected samples completed Hankel/SVD/A/eigendecomposition and both D(t) candidates: D=`5/5`, E=`5/5`. No selected sample had NaN/Inf; eigendecomposition failures=`0`. Every Hankel matrix had shape `64 x 4034`.
- Overall `linearity_error` range was `0.090335–0.212764`.
- Primary Chebyshev features showed sample-level non-overlap in Dmax (D `0.07350–0.11565`, E `0.05840–0.06637`), Dmin (D `0.00884–0.01246`, E `0.01444–0.01737`), and Dstd (D `0.00936–0.01284`, E `0.00670–0.00850`). Dmean ranges overlap: D `0.03043–0.03188`, E `0.03114–0.03252`.
- Raw baseline ranges show amplitude-scale differences but some overlap: raw_std D `40.32–186.07`, E `159.83–535.79`; raw_power D `1,694–35,528`, E `25,597–287,194`; raw_entropy D `3.496–3.675`, E `3.671–3.875`; raw_mean D `-44.61–-0.335`, E `-38.00–7.087`.
- The observed non-overlap is descriptive for this fixed random 5+5 sample. No p-values, cross-validation, or classifier were run. Full Bonn processing and Phase 4 remain unstarted.
- Outputs are under `outputs/bonn_smoke/`: audit, manifest, per-sample metrics, features, status JSON, and all requested figures.

## Still unresolved

A. Exact intrinsic-coordinate definition (`V`, `SigmaV`, or another convention).
B. Numerical implementation and eigenvalue/coordinate correspondence in Eq. (8).
C. Exact attractor/reference construction in the paper.

These choices remain explicit implementation candidates; the Phase 3 smoke test did not revise them.

## Phase 4 — full Bonn D/E validation

**Status: COMPLETE — SUPPORTIVE (segment-level evidence only).**

- All 200 segments completed the frozen Phase 3 pipeline: D=`100/100`, E=`100/100`; no sample failures, NaN/Inf, SVD failures, or eigendecomposition failures.
- D/E linearity error mean ± SD was `0.15814 ± 0.02485` vs `0.14372 ± 0.01798`. Spectral radius mean ± SD was `0.99332 ± 0.00296` vs `0.99421 ± 0.00232`.
- Four of six primary Koopman features had Holm-adjusted Mann–Whitney p < 0.05. Dmax and Dstd retain the Phase 3 direction and sizeable effects, but their full-cohort ranges overlap; the smoke test's complete range separation did not persist.
- Raw `std` and `power` had very strong group effects. Koopman-only CV was below raw-only; combined features exceeded raw-only in mean balanced accuracy and ROC-AUC for all three fixed classifiers, with positive paired-fold differences on most of the 25 shared folds. This is segment-level evidence of complementary information, not proof of a unique mechanism.
- The F094/S026 10× scale control changed the six Koopman features by at most `1.21e-14` relative difference; raw mean/std/power changed as expected. No pipeline setting was selected or changed using these results.
- Full outputs: `outputs/bonn_full/`; report: `RESULTS_PHASE4.md`.

The Bonn result is segment-level only. Reliable patient identity was not established, so it supports no cross-patient, cross-subject, LOSO, or patient-independent generalization claim. Phase 4 is the last phase performed under the current authorization; no SEED experiment was started.
