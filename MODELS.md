# Models and backtest

`uv run scout backtest` runs everything below in under a minute, locally,
with no AWS calls. Results: `reports/backtest_<target>.md` (overall, by
league, by tier, by season) and `reports/backtest_summary.json`; every out of
time prediction goes to `data/processed/models/backtest_<target>.parquet`.

## Targets

All on a log scale, so an error of 0.10 means about 10%, for a 1m player and a
100m player alike. Only regular players (450+ minutes) are modelled.

| Target | Label | Rows | May use current value? |
|---|---|---|---|
| `fair_value` | log of the market value at the feature date | 11,360 | no: it is the target |
| `growth_24m` | log of value 24 months later divided by value now | 8,414 | yes |
| `growth_36m` | the same, 36 months later | 7,233 | yes |

## Expanding window backtest

Every season is predicted by a model trained only on earlier seasons:

* **Gap:** for test season `t`, training uses seasons up to
  `t - max(1, horizon)`. A 24 month label is observed two seasons after its
  row, so training for 2018 stops at 2016.
* **Purge:** a season that ran late can still leak. 2019/20 ended in July
  2020, so one of its 24 month labels was observed after the 30 June 2022
  feature date of season 2021. Such training rows are dropped.
* **Check:** before each fold, `check_no_leak` asserts on the actual dates
  that no training row or label is later than the earliest test row. It
  caught the 2019/20 case above before the purge existed.
* At least three seasons of history before the first test season.

Random k fold is not used: it would let a model learn from a player's later
seasons (and from later price levels) while predicting his earlier ones.

## Models

| Model | What it is |
|---|---|
| `no_change` | growth only: the value stays where it is |
| `group_median` | the baseline: median target of players with the same league, position and age band in the training seasons, falling back to coarser groups below 15 players |
| `lightgbm` | gradient boosted trees on all feature columns, **untuned** defaults (`scout.models.estimators.DEFAULT_PARAMS`) |

## First results (untuned)

| Target | Model | MAE (log) | Median error | Rank correlation |
|---|---|---|---|---|
| fair value | group_median | 0.835 | 60.0% | 0.774 |
| fair value | **lightgbm** | **0.536** | **41.3%** | **0.907** |
| growth 24m | no_change | 0.748 | 66.7% | |
| growth 24m | **group_median** | **0.564** | **42.3%** | 0.429 |
| growth 24m | lightgbm | 0.573 | 43.0% | **0.456** |
| growth 36m | no_change | 1.026 | 115.4% | |
| growth 36m | group_median | 0.710 | 53.2% | 0.442 |
| growth 36m | **lightgbm** | **0.697** | 53.3% | **0.502** |

Out of time predictions: fair value for 2013 to 2025 (9,832 rows), growth
24m for 2014 to 2022 (6,332), growth 36m for 2015 to 2021 (4,655).

What they say:

* **Fair value works.** The tree model halves the gap to the truth compared
  with the baseline and ranks players far better (0.907), in both leagues:
  EPL 0.846, Süper Lig 0.740 rank correlation.
* **Growth is hard, and the baseline is a real competitor.** Untuned trees
  rank slightly better than the age and position baseline but do not beat
  it on error; in the Süper Lig the baseline wins (0.644 against 0.666 MAE).
  Most of what is predictable about growth is age. This is the target where
  tuning, and features about the move itself, have to prove their worth.
* Errors are lower for EPL than for the Süper Lig, in line with the noisier
  Süper Lig values found in FEASIBILITY.md.

## Automatic model tuning (growth 24m)

`scout tune --target growth_24m` ran SageMaker automatic model tuning:
Bayesian search, 20 training jobs, 2 at a time, on `ml.m5.large`. Each job ran
the full expanding window backtest above and reported its mean out of time
error; the tuner minimised it. Wall clock about 37 minutes; 1,769 billable
seconds in total (under half an instance hour), 63 to 149 seconds per job.
Every job reproduced the local numbers exactly when given the same settings.

Best settings (job `scout-tune-g24-1009-063811-004`):

| Setting | Untuned | Tuned |
|---|---|---|
| objective | regression (squared error) | **regression_l1 (absolute error)** |
| learning_rate | 0.03 | 0.0070 |
| num_boost_round | 600 | 568 |
| num_leaves | 31 | 46 |
| min_data_in_leaf | 20 | 67 |
| feature_fraction | 0.8 | 0.88 |
| bagging_fraction | 0.8 | 0.82 |
| lambda_l2 | 1.0 | 0.28 |

| Model | MAE (log) | Median error | Rank correlation | EPL MAE | Süper Lig MAE |
|---|---|---|---|---|---|
| no_change | 0.748 | 66.7% | | 0.634 | 0.877 |
| group_median (baseline) | 0.564 | 42.3% | 0.429 | 0.493 | 0.644 |
| lightgbm (untuned) | 0.573 | 43.0% | 0.456 | 0.491 | 0.666 |
| **lightgbm_tuned** | **0.558** | **42.2%** | **0.461** | **0.484** | **0.642** |

What it says:

* **The loss function matters most.** 18 of the 20 candidates, and the top 11,
  used absolute error. Value changes have heavy tails (a few players jump
  tenfold); squared error chases those, absolute error predicts the typical
  case, which is what the error metric rewards.
* **Tuned trees now beat the baseline**, in both leagues, but only just: 0.558
  against 0.564 overall, 0.642 against 0.644 in the Süper Lig. By season the
  tuned model is better in five seasons, level in two and worse in two
  (2017, 2018).
* **The margin is optimistic.** The settings were picked as the best of 20 on
  these same test seasons, so part of the 1% gain is selection luck. The clean
  check is to tune on test seasons up to 2019 only and score the frozen
  settings on 2020 to 2022.

## More leagues, same features (October 2026)

The data grew from 2 to 17 leagues (the other top four, the Championship,
three feeder leagues, Scotland, Denmark, Greece, Russia, Ukraine, Brazil and
Argentina), the feature code unchanged. Compared on the **same EPL and Süper
Lig test rows** as above (labels checked identical), untuned LightGBM:

| Target | MAE before | MAE now | Rank corr. before | Rank corr. now | Baseline MAE |
|---|---|---|---|---|---|
| fair value | 0.536 | **0.456** | 0.907 | **0.935** | 0.835 |
| growth 24m | 0.573 | **0.550** | 0.456 | **0.498** | 0.564 |
| growth 36m | 0.697 | **0.680** | 0.502 | **0.522** | 0.710 |

More players to learn from helps every target. Untuned trees now beat the
baseline on 24 month growth too, which before only the tuned model did.
Across all 17 leagues, fair value errors are lowest in the big five and the
feeder leagues (about 0.42 to 0.46) and highest in Argentina (0.60) and
Ukraine (0.64), where values are thinnest.

## Next

1. The clean check above: tune on earlier seasons, score on later ones.
2. Error analysis on the saved predictions: which players the models miss,
   and what their seasons looked like before a big jump.
