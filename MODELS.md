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

## Next

1. Automatic model tuning in SageMaker, scoring each candidate on this exact
   backtest, so tuning can't overfit a single split.
2. Error analysis on the saved predictions: which players the models miss,
   and what their seasons looked like before a big jump.
