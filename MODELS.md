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

## Context features (cups, team strength, injuries, transfers)

Same comparison, now between the plain features above (v1) and v1 plus the
context features (v2, FEATURES.md), on test rows whose label is identical in
both (Brazil and Argentina left out: their feature dates moved to January).
MAE in log units and rank correlation, untuned LightGBM:

| Target | Rows | v1 MAE | v2 MAE | v1 rank | v2 rank |
|---|---|---|---|---|---|
| fair value, EPL and Süper Lig | 9,832 | 0.456 | **0.391** | 0.935 | **0.951** |
| fair value, all leagues | 64,780 | 0.461 | **0.392** | 0.903 | **0.930** |
| growth 24m, EPL and Süper Lig | 6,332 | 0.550 | 0.552 | 0.498 | 0.490 |
| growth 24m, all leagues | 40,415 | 0.537 | 0.537 | 0.553 | 0.556 |
| growth 36m, all leagues | 29,674 | 0.658 | 0.661 | 0.589 | 0.586 |

* **Fair value gains a lot**, mostly from transfer fees (16% of the model's
  gain: the last fee and the highest fee so far), team strength (6%) and cup
  and national team minutes (5%). A fee paid before the feature date is
  public information, so this is not a leak, but it means the fair value
  model partly learns what a player last cost.
* **Growth does not move.** Age still carries half of that model; the new
  columns add under 5% between them and no accuracy. What a player did in
  the last year says little about whether his value will jump in the next
  two; that is the question the move model (next) asks more directly.
* Brazil and Argentina, with their January feature dates: fair value MAE
  0.518 (baseline 0.869), growth 24m 0.563 (baseline 0.706).

## Big move model (`scout moves`)

The question from the start of the project: which players at ordinary clubs
will join a top club? Details and the shortlist are in `reports/moves.md`.

* **Top club:** the 20 highest squad values per season over all 17 leagues
  (in 2023: City, Arsenal, Real Madrid, PSG, Liverpool, Chelsea, Bayern,
  Barcelona, ... Brighton, Dortmund, Leipzig, Napoli, Real Sociedad).
* **Rows:** regular players not at a top club at the feature date, and not
  already signed by one (Mastantuono, River Plate 2025, joined Real Madrid
  before the January feature date).
* **Label:** a move (transfer, loan or free) to a top club within 24 months.
  A move back to a club he had left before does not count: in the first
  version a third of the "big moves" were loanees returning to their owner.
* Same expanding window backtest; a label counts as observed when its window
  closes, so open windows are purged from training.

| Model | Rows | Movers | AUC | Average precision | Movers in each season's top 50 |
|---|---|---|---|---|---|
| base rate | 59,344 | 1,385 | | 0.023 | 2.3% |
| group rate (league, position, age) | | | 0.780 | 0.083 | 12.7% |
| **LightGBM** | | | **0.905** | **0.233** | **38.9%** |

* **Not just deals already agreed.** Moves dated 1 July were often agreed in
  spring, before the 30 June feature date. Scored separately: moves in the
  same summer AUC 0.927, the next winter 0.879, a year or more later 0.895
  (baseline 0.77 to 0.79). For moves a year or more later, 17.5% of each
  season's top 50 make one, against 1.1% of all players.
* **Caught in their season's top 50**, among others: Rakitić (Sevilla to
  Barcelona), Fabinho (Monaco to Liverpool), Keïta (Leipzig to Liverpool),
  Havertz (Leverkusen to Chelsea), Dybala (Palermo to Juventus), Nuno Mendes
  (Sporting to PSG), João Neves (Benfica to PSG).
* **A live check.** The newest season (feature date 30 June 2026) has no
  labels yet, but the transfer data runs to October 2026: 9 of the 50 highest
  scored players have already joined a top club (Manzambi to Aston Villa,
  Bouaddi to Manchester City, Godts to PSG, Baleba to Manchester United, ...),
  against 45 of all 6,026 candidates (0.7%).
* AUC is lowest in the big five leagues (0.79 to 0.88), where more players
  move and the question is harder, and highest in the seller leagues (0.91
  to 0.97).

## Next

1. The clean check above: tune on earlier seasons, score on later ones.
2. Error analysis on the saved predictions: which players the models miss,
   and what their seasons looked like before a big jump.
