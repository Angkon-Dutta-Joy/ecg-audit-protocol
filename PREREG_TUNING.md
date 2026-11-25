# Pre-registration of the additional analyses (written before any of them was run)

Date: 2026-11-24. This file is committed to git before the first run; its commit hash and date are the time stamp. It will
also be deposited on OSF by the authors. Everything below is fixed; deviations will be reported as deviations.
The earlier analyses (primary cross-validation, hold-out split, controlled shortcut experiment, recording-context kriging)
keep their own earlier plans and are not changed by this document.

## A. Fair tuning of the proposed model (DDPM+GAT+SVD)
- **Where:** each of the 4 subject-wise cross-validation folds separately (nested). Fitting uses only the fold's training
  participants; every selection uses only the fold's 3 validation participants. The fold's test participants are not
  touched until the winner is fixed. The folds are rebuilt from the raw files and must be bit-identical (split,
  evaluation windows, masks, validation windows, normalisation, LRMC rank) to the original cross-validation run.
- **Configurations (fixed, no others will be added):** c0 = the pre-specified configuration (lr 1e-4, dropout 0.05,
  weight decay 1e-4); c1 lr 2e-4; c2 lr 5e-5; c3 dropout 0.2 + weight decay 1e-2; c4 dropout 0.2 + weight decay 1e-2 +
  lr 2e-4. Architecture, diffusion schedule (T = 300), batch size 16, 50-epoch budget and data are unchanged.
- **Successive halving:** all 5 configurations to 10 epochs; the best 2 continue to 20; the best continues to 50. Every
  run uses the cosine learning-rate schedule of the full 50-epoch budget, so continuing a run equals training it at once.
- **Selection metric (ranking and checkpoint):** masked-entry MSE of the mean of 2 diffusion samples on 128 fixed
  validation windows with fixed masks and fixed sampling noise, evaluated every 2 epochs; ranking uses the best value
  reached so far; the checkpoint with the lowest value is the tuned model.
- **Test evaluation:** the fold's frozen test windows and primary masks, identical sampling seeds to all other models;
  sample 0 = single-sample result, mean of 5 samples = multi-sample result.
- **Reporting:** both models are reported: the pre-specified model (configuration c0, checkpoint by validation loss) and
  the tuned model (as above). Nothing is replaced. The selected configuration and best epoch of every fold are reported.
- **Comparison and decision rule:** tuned model vs window LRMC, per participant (N = 32), two-sided exact Wilcoxon
  signed-rank test on MSE and PCC (primary analysis family of the paper, Holm correction within metric), with
  Hodges-Lehmann paired difference and rank-biserial correlation. The tuned model is described as "more accurate than
  LRMC" only if its MSE is lower for at least 24 of 32 participants AND the Holm-adjusted Wilcoxon p < 0.05; "comparable"
  if neither method is lower for 24 or more participants; otherwise "less accurate".
- **Title rule:** a title that presents the model as more accurate than LRMC is used only if the rule above gives "more
  accurate"; otherwise the neutral title is kept.

## B. Multi-sample evaluation of the diffusion models
- Posterior-mean estimate = mean of 20 samples per window on the cross-validation test windows (pre-specified and tuned
  models) and of 50 samples on the hold-out test set (pre-specified model). Sampling seeds fixed in advance
  (sample_seed * 100003 + batch offset + 7919 * k, as for all earlier samples).
- Probabilistic score: CRPS per missing entry estimated from the samples (energy form, unbiased with the sample spread),
  averaged per participant. Interval coverage of the central 90 % sample interval. Deterministic methods are scored by
  CRPS = absolute error (a point forecast) and are labelled as such.

## C. SAITS baseline
- PyPOTS implementation of SAITS with its default hyper-parameters, input = the observed entries of the 32 x 640 window
  (time steps x features), trained per fold on the same training windows and masks generator as the diffusion models;
  early stopping and checkpoint by masked-entry MSE on the fold's validation windows; then the fold's frozen test set.
  If the default model does not fit a T4 GPU, the window is processed at the same resolution in two halves; any such
  change is reported.

- **Addendum (2026-11-24, before any SAITS run):** the PyPOTS 2.0 SAITS constructor has no default for its architecture,
  so the settings of the PyPOTS documentation example are used: 2 layers, d_model 256, 4 heads, d_k = d_v = 64,
  d_ffn 128, dropout 0.1, attention dropout 0.1; PyPOTS defaults otherwise (batch 32, up to 100 epochs, Adam, MAE
  training loss, validation MSE), early-stopping patience 10. Training masks: the primary gap masks (seed 42 + 21);
  validation masks: the masks used for the LRMC rank selection. Predictions are also stored for the robustness
  conditions (2-s/1-s gaps, four whole channels) of the cross-validation folds. Implementation: src/ojemb/saits_baseline.py.
- **Addendum 2 (2026-11-24, before any run of this variant; after fold 0 of the pre-registered SAITS had finished):**
  an independent code audit noted that PyPOTS' SAITS learns from randomly hidden single points (MCAR, 20 % of observed
  entries) and that the pre-registered setup kept one fixed gap mask per training window, whereas all methods are scored
  on 77-sample gaps and the diffusion models train on freshly drawn gap masks. A sensitivity variant, "SAITS
  (gap-trained)", is therefore added and reported NEXT TO the pre-registered SAITS (which is completed and reported
  unchanged): training windows fully observed; the artificial training masks are drawn from the study's gap protocol,
  freshly for every sample at every fetch (seed 42 + 31); the masked-imputation loss is computed on these gap entries;
  everything else identical to the pre-registered SAITS (architecture, optimiser, early stopping on the same validation
  windows and masks, same test windows and conditions). No other variant will be added. Implementation:
  src/ojemb/saits_gap.py; tests: tools/smoke_saits_gap.py.
- **Deviation (2026-11-24, recorded before the gap-trained run started; no SAITS test result had been analysed):** the
  pre-registered SAITS configuration is discontinued after fold 0 because its training task (random single points) does
  not match the evaluation task (77-sample gaps), as found by the audit above. The gap-trained configuration replaces it
  as THE SAITS baseline for all 4 folds. The completed fold 0 of the pre-registered configuration is kept unchanged
  (MyDrive/ojemb_saits/state) and reported in the Supporting Information; the incomplete fold 1 is not used. The paper
  states this deviation and its reason.

## D. Size-matched window-level cell of the shortcut experiment
- Same as cell C (window-level split, observed-only prior, 1,200 optimiser steps per epoch, 50 epochs), but the training
  pool is a uniformly random subset (seed 42) of 19,200 windows of the window-level pool, i.e. the size of the
  subject-wise pool. Three training seeds. Reported as the geometric-mean MSE ratio to window LRMC and the number of
  the 8 test participants with ratio < 1, next to cells A and C.

## E. Replication on an openly licensed dataset (PhysioNet EEG Motor Movement/Imagery, ODC-By 1.0)
- Participants: all 109, except any whose recordings are not sampled at 160 Hz or lack runs (rule on data properties,
  applied before any model is run; the excluded IDs are reported).
- Channels: the 32 channels of the DEAP montage (all present in this dataset). Preprocessing to match DEAP: band-pass
  4-45 Hz (zero-phase 4th-order Butterworth), resampling 160 -> 128 Hz (polyphase), common average reference over the
  32 kept channels, per-run mean removal, z-scoring and 1st/99th-percentile clipping with training-participant statistics.
- Windows: all 14 runs per participant; 5-s windows, training stride 2.5 s, evaluation windows non-overlapping, 60 fixed
  evaluation windows per test participant; identical masking protocol.
- Design: subject-wise 4-fold cross-validation (fixed seed); per fold 3 validation participants. Methods: mean, linear,
  spherical spline, window LRMC (rank selected per fold on validation), recording-context kriging (gamma, lambda selected
  per fold on validation; recording = all runs of the participant), the proposed model with the configuration selected
  most often across the DEAP folds (ties: lower mean validation score), SAITS as in C. 1,200 optimiser steps per epoch.
- **Addendum (2026-11-24, after preprocessing, before any method was run on this dataset):** each run is cut into
  consecutive 60-s segments ("trials", 7,680 samples at 128 Hz): R01/R02 give 1, R03-R14 give 2, i.e. 26 per participant.
  A run too short to give its segments is treated as a missing run. Applying the rule excluded S088, S092, S100 (sampled
  at 128 Hz) and S104, S106 (one run too short); 104 participants are included (tools/prep_eegmmidb.py, manifest in
  results/eegmmidb/manifest.json; all 1,526 EDF files match PhysioNet's SHA-256 checksums).
- "Replicated" means: the direction of each primary comparison (model vs LRMC, kriging vs LRMC) agrees with DEAP and the
  Holm-adjusted Wilcoxon p < 0.05.

## F. Statistics added to all participant-level comparisons
- The pre-specified participant-level Wilcoxon tests remain primary. The paired t-test is no longer reported.
- Sensitivity analysis: linear mixed model of the participant log MSE ratio (method vs LRMC) with a random intercept per
  cross-validation fold.
- Effect sizes: Hodges-Lehmann paired difference with 95 % CI, rank-biserial correlation; for N = 8, exact binomial
  (Clopper-Pearson) CI of the proportion of participants favouring a method.

## G. Order and cut line
Order of execution: A, B, C, E, D, then the cheap sensitivity checks. Anything not finished by the cut-line date
(day 11 of the plan) is reported as not done, with the reason. Never cut: A, B, C, E.

## Addendum 3 (2026-11-26, before any run of the analyses below)

1. **Cut-line.** Section G set the cut-line as "day 11 of the plan" but never defined day 1. It is fixed now:
   6 December 2026, 23:59 Bangladesh time (17:59 UTC). Anything unfinished then is reported as "not done, with the
   reason" and listed as a deviation. It can be moved only by a dated addendum committed before that time.

2. **Section E, proposed model.** Configuration c4 (dropout 0.2, weight decay 1e-2, lr 2e-4), chosen by the tie rule of
   section E (c1 and c4 each won two DEAP folds; mean validation score c4 0.0803 < c1 0.0809). 50 epochs of 1,200
   optimiser steps (batch 16, shuffled training pool), cosine learning-rate schedule over 50 epochs, no early stopping.
   Checkpoint: lowest validation imputation MSE as in section A (128 fixed validation windows from the fold's 3
   validation participants, fixed masks and sampling noise, mean of 2 samples, scored every 2 epochs). Test: the fold's
   frozen test windows, primary masks, the same sampling seeds as every other model; sample 0 = single-sample result,
   mean of 5 samples = multi-sample result.

3. **Section B estimators.** CRPS per missing entry, fair (unbiased) form:
   CRPS = (1/n) sum_k |x_k - y| - 1/(2n(n-1)) sum_{k != l} |x_k - x_l|, averaged per participant; deterministic
   methods: CRPS = absolute error (labelled as point forecasts). Interval: the central interval between the order
   statistics X_(i) and X_(n+1-i), with i chosen so that the nominal coverage (n+1-2i)/(n+1) is closest to 0.90:
   n = 20 -> (1, 20), nominal 19/21 = 0.905; n = 50 -> (3, 48), nominal 45/51 = 0.882. ("Central" follows section B;
   without it, n = 50 would allow asymmetric pairs with 46/51 = 0.902.) Observed coverage is reported next to the
   nominal value.

4. **Holm family (within metric).** DEAP: pre-specified model vs LRMC; pre-specified model vs DDPM; tuned model (single
   sample) vs LRMC. Secondary, reported with unadjusted exact p-values and effect sizes: multi-sample means, SAITS vs
   LRMC, SAITS vs tuned model. EEGMMIDB: model vs LRMC and kriging vs LRMC (section E). The Holm adjustment in
   results/tuning/tuned_vs_lrmc_stats.csv (two comparisons) is replaced by this family.

5. **Compute and accounts.** All gap-trained SAITS runs (the 4 DEAP folds first, then the 4 EEGMMIDB folds) are Kaggle
   batch sessions ("Save Version -> Save & Run All (Commit)", which run on Kaggle's servers without an open browser) on
   a Tesla T4, on one co-author's Kaggle account under the co-author's own login; no password or token is shared. Each
   fold is one notebook generated by script from the tested code, with no cell to edit; it checks the GPU type, the
   code commit hash, the data-file hashes and the hashes of split, windows and masks at start and stops if any differs;
   it saves a checkpoint after every epoch and resumes from the previous output if a session ends early. Everything
   that uses the diffusion model (section B, the EEGMMIDB smoke test and model, section D, exploratory S2) and all CPU
   jobs run on the first author's Kaggle account. Every run records the code commit hash, the data-file hashes, the
   GPU type and the account. Sanity check, not a stop rule (validity does not depend on it, because all SAITS folds run
   on the same account): the first epoch of DEAP fold 0 is also run on the first author's account, and the two
   validation MSE values "agree" if they differ by at most 1 % of the first author's value. Reason: same code, data,
   seeds and GPU type, so only non-deterministic GPU arithmetic and library versions can differ; these are expected to
   change the value far less than 1 %, while the differences between methods that the paper interprets are tens of
   per cent. A larger difference is reported to the first author with both values, and the first author decides.
   Colab is no longer used; fold 0 of the pre-registered SAITS, finished on Colab, is kept unchanged.

6. **Contingency for section B.** If the cross-validation checkpoints of the pre-specified model cannot be recovered,
   section B for that model is reported for the folds that exist and for the hold-out split. No retraining. This is then
   a reported deviation.

7. **Exploratory analyses** (labelled exploratory; no selection of any kind on them).

   **S1, how much recording kriging needs.** For each evaluation window, mean and covariance are estimated as in the
   main analysis (pairwise, observed samples only; the other windows carry the same seeded masks as in the main
   analysis; lambda fixed at the selected 0.01) but from a set of k windows of the participant's non-overlapping
   640-sample grid. The set always contains the target window itself: k = 1 means the target window's own observed
   samples only (window-level information). Units are the stored 60-s trials (DEAP: 40 trials; EEGMMIDB: 26 segments,
   in the order R01, R02, R03 first and second minute, ..., R14 second minute). Ordering for "nearest": first the
   windows of the target's own trial/segment by absolute start difference from the target (ties: earlier window); then
   the other trials/segments, ordered by (a) EEGMMIDB only: the other segment of the same run first, (b) absolute
   difference of stored index from the target's (ties: lower index), each contributing its windows from its start
   onward, until k windows are reached. k = 1, 3, 12, 60 and all (DEAP 480, EEGMMIDB 312 windows). k = 12 is the
   target's whole trial/segment, i.e. the "same trial" scope (gamma = 0) of the main analysis; k = all is the
   participant scope (gamma = 1) and must reproduce the main-analysis numbers exactly (check). Reported per k as MSE and
   number of participants better than window LRMC, DEAP cross-validation and EEGMMIDB.
   *Temporal order:* within a trial/segment the order is true recording time. Across trials it is NOT verified
   recording order. DEAP: our loader reads the 40 trials in stored file order and does not re-order them; the data
   mirror contains no metadata file that maps trials to presentation order; the official documentation page could not
   be opened on 25-26 November (TLS error), and a third-party preprocessing pipeline re-orders trials "to follow the
   experiment_ID order", which suggests, without confirming, that stored order is video order rather than presentation
   order. EEGMMIDB: PhysioNet states that the first two runs are the baselines and the other 12 the task runs, but not
   the order of the task runs; the EDF header start times are the same placeholder value in every run of all 109
   participants (checked 26 November), so recording order cannot be confirmed; within a run the two 60-s segments are
   consecutive. k = 60 therefore means "the target trial/segment plus its neighbours in stored order", not "the nearest
   minutes of recording".

   **S2, all channels missing at once.** One 0.6-s gap (77 samples) at the same position in all 32 channels of a
   window. For cross-validation fold f of each dataset, rng = numpy.random.default_rng([2612, f]) draws one start per
   evaluation window, in the order of the fold's frozen evaluation index, uniformly from the integers 180..323 (the
   primary range). Frozen evaluation windows of the DEAP cross-validation and of EEGMMIDB; methods mean, linear, window
   LRMC, kriging (settings unchanged), tuned model and gap-trained SAITS with their existing checkpoints, sample 0. No
   retraining.

   **Results already known on this date:** DEAP primary cross-validation, hold-out and shortcut-experiment results;
   recording-context kriging on DEAP (all four conditions); section A tuning and its test comparison with LRMC;
   validation MSE of the pre-registered SAITS on DEAP fold 0 (its test outputs exist but have not been analysed).
   EEGMMIDB: no method run, except a pipeline check (24 November) that printed, for fold 0, the LRMC rank chosen on
   validation (4) and one test participant's masked MSE (S001: kriging 0.113, mean imputation 0.767).

## Addendum 4 (2026-11-26, before any section B sample is drawn or scored)
**Averaging of CRPS and interval coverage** ("averaged per participant" in section B was not specific). Both follow
exactly the path of the masked-entry MSE in the Methods (`metrics.pointwise -> window_mean -> subject_mean`): the
per-entry CRPS (or the 0/1 indicator that the entry lies inside the central interval) is averaged over the gap
entries of each (window, channel) pair that has at least 10 missing samples (pairs with fewer are not scored), then
over the scored channels of the window, then over the participant's windows. Deterministic methods (CRPS = absolute
error) are averaged on the same path. The nominal coverage of the interval is reported next to the observed value.

## Addendum 5 (2026-11-26, after the EEGMMIDB evaluation data were frozen (commit a45b25a), before any method is run on EEGMMIDB)
**Section E, classical baselines: settings not listed in section E, fixed to exactly those used on DEAP.**
- Methods: mean imputation, linear interpolation, spherical spline (`baselines.CLASSICAL`), window LRMC
  (`baselines.lowrank_complete`, 100 iterations), recording-context kriging (`replication.kriging`, the function that
  produced the DEAP results).
- Selection on the fold's 3 validation participants only, with the primary-protocol masks drawn from
  default_rng(seed + 3) (as on DEAP): LRMC rank from {2, 3, 4, 6, 8} (`runner.select_rank`); kriging (gamma, lambda)
  from gamma in {0, 0.25, 0.5, 1} x lambda in {1e-3, 1e-2, 1e-1}, minimum validation masked MSE (gamma = 0: the
  target's 60-s segment; gamma = 1: all segments of the participant = all runs).
- Evaluation: the frozen test windows and primary masks of each fold; per participant MSE, PCC and NMSE through
  `replication.per_subject` (the DEAP path). Robustness conditions are not part of section E and are not run here.
- Statistics: section F as on DEAP; EEGMMIDB Holm family: model vs LRMC and kriging vs LRMC (addendum 3, item 4).

## Erratum (2026-11-25, 19:5x Bangladesh time)
The headings of addenda 3, 4 and 5 say "2026-11-26", and addendum 3 item 7 says "checked 26 November" and "25-26
November". The correct date of all three addenda and of those checks is **25 November 2026** (Bangladesh time, UTC+6);
the date was copied from the execution plan's schedule instead of the clock. The git commits are the time stamps:
addendum 3 66c6b2f (17:34), addendum 4 a1374e3 (18:21), addendum 5 828ebef (19:14), all 2026-11-25 +0600. The order of
rules and runs is unaffected (addendum 4 before the first section B run; addendum 5 after the EEGMMIDB evaluation data
were frozen in a45b25a (19:03) and before the first EEGMMIDB method run). The addenda text is otherwise unchanged.

## Clarification (2026-11-25, after an independent review of today's work)
Addendum 5 says it was written "before any method is run on EEGMMIDB". Precisely: before any method was run on EEGMMIDB
*as an analysis*. The 24-November pipeline check disclosed in addendum 3, item 7 (LRMC rank selection on fold-0
validation participants, rank 4; kriging and mean imputation on one test participant, S001) had run earlier. Addendum
5 copies the DEAP settings unchanged, so that check gave no room for choice; the paper will not claim that no method had
been run on EEGMMIDB before the analysis.
