# Results: Next-Day Cross-Sectional Return Prediction (TW-50)

Universe `tw50` (30 US / 50 TW, 14.0% ADR pairing). Walk-forward split,
246 test days. IC = mean daily cross-sectional Pearson correlation between
predicted and realised next-day log return; RankIC = Spearman equivalent.
All numbers produced by `scripts/paired_daily.py` and `scripts/mktable`-style
aggregation over `runs/`.

> **Numbers revised 2026-08-18 (MPS non-determinism defect).** Every neural
> row below was re-evaluated on CPU from its own `best.pt`. The previously
> published figures were computed on MPS, where this model's evaluation is
> not reproducible — the same checkpoint re-evaluated five times spanned
> test IC +0.0298 to +0.0466 (range 0.0169), against five bit-identical
> +0.058480 on CPU. See [Reproducibility defect](#reproducibility-defect-and-correction)
> for what changed and what it does and does not fix.

## Main comparison

Rows are sorted by test IC, descending. The reference model for all
statistical tests is **MAGNET-v2** (row 7), not the top row.
`Δ IC` is *reference minus row*, so a positive value means MAGNET-v2 is better.
`p (seed)` is a paired t-test over matched seeds; `p (day)` is a paired
t-test over the 246 daily IC values (seed-averaged series).

| Model | Category | Seeds | IC | sd | RankIC | sd | Δ IC | p (seed) | p (day) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ridge, US+TW, rank target | Linear baseline | 1 | +0.1077 | — | +0.1095 | — | −0.0602 | — | **<0.001** |
| Bipartite screen + LASSO [24] | Linear baseline | 1 | +0.1039 | — | +0.0921 | — | −0.0564 | — | **<0.001** |
| Bipartite screen, ens-avg [24] | Linear baseline | 1 | +0.1033 | — | +0.0899 | — | −0.0558 | — | **<0.001** |
| Ridge, 30 US t-1 returns | Linear baseline | 1 | +0.1020 | — | +0.0961 | — | −0.0545 | — | **0.002** |
| Bipartite screen, ens-med [24] | Linear baseline | 1 | +0.0984 | — | +0.0873 | — | −0.0509 | — | **<0.001** |
| Early fusion (T=20, F=9, L=2) | Neural baseline | 10 | **+0.0518** | 0.0049 | +0.0412 | 0.0040 | −0.0043 | 0.481 | 0.718 |
| **MAGNET-v2 (this work)** | This work | 10 | +0.0475 | 0.0180 | **+0.0513** | 0.0129 | — | — | — |
| MAGNET (F9, T=1, L=1) | This work | 10 | +0.0259 | 0.0059 | +0.0407 | 0.0040 | +0.0216 | **0.002** | **0.012** |
| MAGNET-v2 w/o skip (ablation) | This work | 10 | +0.0241 | 0.0090 | +0.0342 | 0.0071 | +0.0234 | **0.010** | **0.005** |
| Constant (train-mean rank) | Null model | 1 | +0.0149 | — | +0.0317 | — | +0.0326 | — | 0.017 † |
| MAGNET (original: F9, T=20, L=2) | This work | 3 | +0.0128 | 0.0017 | +0.0326 | 0.0024 | +0.0414 | **0.048** | **0.005** |
| MAN-SF [11] | Neural baseline | 3 | +0.0113 | 0.0081 | +0.0071 | 0.0081 | +0.0429 | 0.073 | **0.006** |
| MEIG [1] | Neural baseline | 3 | +0.0061 | 0.0130 | +0.0156 | 0.0051 | +0.0481 | 0.093 | **0.004** |
| MAGNET-intermediate | This work | 3 | +0.0030 | 0.0040 | −0.0044 | 0.0081 | +0.0512 | **0.014** | **0.002** |
| LSTM only (no graph) | Neural baseline | 3 | +0.0012 | 0.0051 | +0.0139 | 0.0056 | +0.0530 | **0.028** | **0.005** |
| HGT [14] | Neural baseline | 3 | +0.0011 | 0.0108 | +0.0075 | 0.0036 | +0.0531 | **0.005** | **0.004** |
| Adv-ALSTM [10] | Neural baseline | 3 | −0.0014 | 0.0039 | +0.0043 | 0.0014 | +0.0556 | **0.015** | **0.001** |
| DeltaLag [13] | Neural baseline | 3 | −0.0014 | 0.0105 | −0.0056 | 0.0043 | +0.0556 | **0.035** | **<0.001** |
| HATS [12] | Neural baseline | 3 | −0.0053 | 0.0047 | +0.0092 | 0.0066 | +0.0595 | **0.033** | **<0.001** |
| Early fusion (T=1, F=3, L=1) | Neural baseline | 10 | −0.0076 | 0.0034 | +0.0006 | 0.0083 | +0.0550 | **<0.001** | **0.006** |

Rows with 3 seeds compare against MAGNET-v2 restricted to the same 3 seeds
(42/7/123), on which MAGNET-v2 averages +0.0542; the `Δ IC` for those rows is
therefore relative to +0.0542, not to the +0.0475 shown in the table.

† The Constant row's previously published `p (day)` of 0.012 could not be
reproduced by the HAC estimator in `scripts/paired_daily.py`: re-running that
estimator on the *old* predictions gives 0.025, so the original figure came
from some other path. The 0.017 shown is the HAC value against the corrected
reference. Treat this single cell as method-ambiguous; every other `p (day)`
in the column was verified to reproduce its published value exactly on the old
predictions before being recomputed.

## Definitions of the non-neural rows

These rows are not literature architectures. They are probes we built to
establish what a simple estimator can extract from this data, and they set the
ceiling that the neural models are measured against. All use ridge regression
with **per-target coefficients**: the design matrix is shared across the 50 TW
stocks, but each stock gets its own coefficient vector. The penalty `alpha` is
chosen on validation cross-sectional IC, and standardisation uses training
statistics only.

| Row | Design matrix X | Fitting target | Free params |
|---|---|---|---:|
| Constant (train-mean rank) | none | — | 0 |
| Ridge, 30 US t-1 returns | 30 US log returns at *t*−1 | TW log return at *t* | 1,500 |
| Ridge, US+TW, rank target | 30 US **+ 50 TW** log returns at *t*−1 (80 cols) | daily cross-sectional **rank** of TW return, standardised | 4,000 |
| Bipartite screen + model [24] | US predictors surviving a rolling univariate t-test screen (w = 250, refit every 10 days, threshold 2) | TW log return at *t* | varies |

**Constant (train-mean rank).** The true null model for a cross-sectional
ranking task: order the 50 TW stocks once by their training-period mean return
and reuse that order every day. The prediction is constant over time but varies
across stocks, which is why it has a well-defined IC. Any model that does not
beat it has learned nothing about the cross-section.

**Ridge, 30 US t-1 returns** (`R2` in `scripts/ridge_ladder.py`). Predicts
`y_hat[t, j] = sum_i W[i, j] * r_US[t-1, i]`. Column `W[:, j]` is TW stock *j*'s
exposure vector over the 30 US names, so the loadings are stock-specific. The
input is 30 scalars per day: no technical indicators, no look-back window, no
graph.

**Ridge, US+TW, rank target** (`KTW+` in `scripts/factor_vs_graph.py`). Adds two
things to the row above. (i) The design matrix also contains the previous day's
returns of all 50 TW stocks — this is *cross-sectional* information about the
other names on the same day, not a stock's own history. A stock's own history
was tested separately (`R0`, 20 days x 9 features) and degenerates to the
constant model. (ii) The target is replaced by the within-day cross-sectional
rank of the realised return, standardised to zero mean and unit variance, which
aligns the training objective with the IC/RankIC evaluation and removes the
influence of the fat tails in raw returns (excess kurtosis 3.53).

**Bipartite screen + model [24].** Our reproduction of the cited method: a
rolling-window pairwise univariate t-statistic selects which US stocks may
predict each TW stock, and the surviving predictors are fed to a suite of models
(`ens-avg` and `ens-med` are the mean and median across models). Reported under
the same walk-forward split as everything else. At this problem size the
screening retains 92% of candidate pairs, and the whole pipeline is statistically
indistinguishable from the plain per-target ridge above.

These four settings span +0.0984 to +0.1077 and are mutually indistinguishable
(pairwise p = 0.23–0.62 across 16 arms tested). We therefore treat +0.10 as a
plateau rather than as the score of any single method.

## MAGNET-v2 versus early fusion (10 seeds each, matched seed set)

Early fusion concatenates the paired ADR features at the input layer and has no
cross-layer coupling or gate. It is the single most important comparison in the
table because it is the simplest model that also uses both markets.

| Metric | Early fusion | MAGNET-v2 | Δ (v2 − EF) | 95% CI | Welch | Paired t | Mann-Whitney | Wilcoxon | v2 wins |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| IC | +0.0518 | +0.0475 | −0.0043 | [−0.0177, +0.0090] | 0.477 | 0.481 | 0.678 | 0.492 | 4/10 |
| RankIC | +0.0412 | +0.0513 | +0.0101 | [+0.0011, +0.0191] | **0.038** | **0.032** | 0.104 | **0.049** | 7/10 |

The two models are **not separable on IC**, with early fusion's point estimate
still higher. The correction narrowed the gap from −0.0079 to −0.0043 and moved
MAGNET-v2 from 2/10 to 4/10 seed wins, but it did not change the conclusion:
p = 0.48, and the CI still spans zero comfortably.

MAGNET-v2 is better on RankIC at the 5% level under the two parametric tests
and Wilcoxon (p = 0.049, i.e. exactly at the boundary), but Mann-Whitney moved
the wrong way under correction — from p = 0.064 to **p = 0.104** — and the CI
lower bound is +0.0011. The RankIC claim is weaker after correction, not
stronger, and now rests on tests that disagree with each other.

Early fusion does not benefit from the input regime tuned for MAGNET-v2: under
T = 1, F = 3, L = 1 it collapses to IC −0.0076 (p < 0.001 against its own
T = 20, F = 9, L = 2 setting). The two architectures therefore require different
input regimes, and the row above compares each at its own best configuration.

## Ablation of the proposed component

MAGNET-v2 adds one component to the tuned MAGNET: a concatenated raw-feature
skip connection into the cross-market coupling point, with feature-wise
(cross-node) normalisation. 10 seeds each, matched seed set.

| Metric | w/o skip | with skip | Δ | 95% CI | Welch | Paired t | Mann-Whitney | Wilcoxon | Wins |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| IC | +0.0241 | +0.0475 | +0.0234 | [+0.0070, +0.0399] | **0.003** | **0.010** | **0.006** | **0.020** | 8/10 |
| RankIC | +0.0342 | +0.0513 | +0.0172 | [+0.0042, +0.0301] | **0.002** | **0.015** | **0.006** | **0.037** | 7/10 |

This is the one comparison the correction **strengthened**: Δ IC rose from
+0.0191 to +0.0234, the CI lower bound moved off zero (+0.0030 → +0.0070), and
Wilcoxon went from borderline (0.049) to 0.020. The skip ablation is the most
robust result in this table.

Mechanism: information at the coupling point that a per-target linear probe
can extract rises from +0.0448 (without skip) to +0.0693–0.0754 (with skip);
the raw-feature ceiling is +0.0874. Five earlier interventions left this
quantity unchanged.

## Notes and limitations

1. **MAGNET-v2 does not beat early fusion on IC.** At 10 seeds each, early
   fusion reaches +0.0518 against MAGNET-v2's +0.0475 and wins on 6 of 10 seeds;
   the difference is not significant (p = 0.48). MAGNET-v2 is ahead on
   RankIC (+0.0101, Welch p = 0.038, Mann-Whitney p = 0.104). The honest summary
   is a split decision, not a win. Any claim that the multiplex coupling and
   gate are necessary must be qualified accordingly: an input-layer
   concatenation with no graph coupling matches or exceeds it on IC.
2. **All linear baselines beat every neural model by a wide, significant
   margin** (Δ ≈ −0.051 to −0.060, p ≤ 0.002). The gap is not closed by this
   work; MAGNET-v2 reaches roughly 44% of the linear plateau.
3. **Baselines are not tuned.** Each literature baseline was run with one
   default configuration, whereas MAGNET variants were swept over ~50
   configurations. This favours the proposed model and must be disclosed.
4. **DeltaLag is not a faithful reproduction** (near-constant predictions on
   some seeds); it is reported for completeness, not as evidence about the
   method.
5. **Three-seed rows are indicative only.** With sigma_seed = 0.0062, the
   minimum detectable difference at n = 3 is 0.0188 and a non-parametric test
   cannot reach p < 0.05 at all (minimum two-sided p = 0.10). Effects estimated
   from 3 seeds are inflated: measured exaggeration factor 1.41x on this
   project's own data.
6. `p (day)` treats days as the unit of replication with the seed set fixed;
   it is more sensitive only when the two models' daily IC series are
   correlated (r_day > 0.7), which holds within an architecture family but
   not across families.

7. **Seed variance is the dominant source of uncertainty, and MAGNET-v2 has
   the worst of it.** Its seed sd (0.0180) is twice that of the ablation arm
   (0.0090) and nearly four times early fusion's (0.0049). The 10-seed sample
   spans +0.0178 to +0.0707. This was true before the correction and is not
   caused by it.

---

## Reproducibility defect and correction

**What was wrong.** Runs before 2026-08-18 computed their evaluation metrics on
MPS (Apple Metal). `GATv2Conv` aggregates neighbours with a scatter-add, and the
MPS backend does not fix the floating-point summation order between runs. The
recorded `meta.json` / `predictions/test_predictions.csv` are therefore a single
draw that not even the same machine reproduces.

Measured on `tw50_T1F3bnl1_s11`, five evaluations of one unchanged `best.pt`:

| device | test IC across 5 identical runs | range |
|---|---|---:|
| CPU | +0.058480, +0.058480, +0.058480, +0.058480, +0.058480 | **0.000000** |
| MPS | +0.046623, +0.036231, +0.031759, +0.035888, +0.029725 | **0.016899** |

The MPS range (0.0169) is the same size as the entire between-seed sd (0.0180),
so it was invisible against seed noise. The recorded value for that run
(+0.0380) sits inside the MPS range.

**What was not wrong.** The weights and the data are intact, and this was
verified rather than assumed:

- `save_checkpoint` / `load_checkpoint` round-trip bit-exactly (state_dict
  compared tensor by tensor after reload: zero differing entries).
- The ground-truth `y` column in every stored CSV matches a fresh forward pass
  to 1e-16, so the dataset, graph snapshots and feature files are unchanged.
- CPU evaluation is deterministic across repeated runs and independent of
  batch size (32/8/16/64/246 all bit-identical).
- CPU and MPS agree to 4.5e-08 on a *single* pass over all 246 days; the
  instability is run-to-run, not a systematic cross-device offset.

**Which arms were affected.** Only architectures that use `GATv2Conv`. Arms
without scatter-based aggregation reproduced their recorded metrics to under
5e-5 — early fusion (both settings), LSTM-only, HATS, Adv-ALSTM and MEIG are
bit-stable. This is independent confirmation of the mechanism.

**The fix.** `training.eval_device` (new, defaults to `cpu`) routes every
`evaluate()` call to a deterministic device while training stays on
`training.device`; CPU evaluation costs 1.3 s per split. Each run now reloads
its own `best.pt` after training and fails loudly if the recorded test IC is
not reproduced to within 1e-4, and `meta.json` records `eval_device` and
`test_IC_roundtrip`. Three regression tests in `tests/test_train.py` lock the
invariant (they fail on MPS, as intended).

**What the correction does and does not fix.** Re-evaluation corrects the
*reported* metrics for the checkpoints that were actually selected. It does
**not** correct the *selection*: best-epoch was chosen on MPS-computed
validation IC, so which epoch got saved was itself decided under noise of the
same magnitude. Numbers here are therefore reproducible but still conditioned
on a noisy model-selection path. Fully clean figures require re-running the
arms with `eval_device: cpu` from the start. Treat the ranking of closely
spaced rows (anything within ~0.005) as provisional until then.

**Reproducing.** `scripts/reeval_checkpoints.py --all-tw50` regenerates
`meta_reeval.json` and `predictions/test_predictions_reeval.csv` for every run
and writes `docs/reeval_summary.csv`; `scripts/paired_daily.py --predictions
reeval` runs the paired tests on the corrected series. Original files are left
untouched for audit.

**Runs that could not be re-evaluated (16).** All are outside this table: the
pre-E6 `tw50chk_wfree` runs (checkpoints predate `pair_src`/`has_pair` becoming
non-persistent) and the twelve superseded LayerNorm-era `tw50_T1F3skip*` /
`tw50_T1F3cat*` runs that commit `b0f4fbd` had already invalidated. Separately,
`tw50_T1F9csdmboth_s7` now yields a NaN IC on re-evaluation (constant
predictions); that arm is also not in this table but the run is suspect.
