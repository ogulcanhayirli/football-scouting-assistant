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

## Ability score version 2 (decision 8, `scout ability`)

The ability score feeds the scouting models (as columns) and the bargain and
performance lists. Version 1 is the API match rating with the league effect
(from players who changed leagues) and team goal difference taken out.
Version 2, built on 19 October 2026 on all 37 leagues:

* **Base:** a per 90 composite per position group instead of the match
  rating. Its measures and weights come from the step up trait comparison:
  the per 90 stats on which players who made the step up stood at least 5
  percentiles above same age peers of their position, weighted by that gap
  (centre backs: goals, passes, shots, duels won; wingers: dribbles,
  assists, key passes, goals, shots; full table in `reports/ability.md`).
  Each stat is standardised within position group and season. The API
  leaves a zero count empty, so a missing count reads as 0 where the league
  season records it.
* **League term:** transfer based, refitted on the composite. **Team term:**
  the club's Elo (mean pre match rating over its league matches, against
  its league's average), refitted on the same player pairs.
* **Two seasons:** this season and the previous one pooled by minutes, then
  shrunk towards the position mean by empirical Bayes. The prior strength
  is fitted per position group from how much a season's score scatters at
  different minutes: 130 minutes for attacking midfielders up to 600 for
  centre backs and 2,100 for goalkeepers (a goalkeeper's per 90 numbers say
  little about him).

**Stability** (season to season Pearson of the ability percentile, same
players, regular in both seasons). The 17 league rows are the pairs of
`reports/elo.md` and reproduce its numbers:

| Version | 17 leagues (32,932 pairs) | 37 leagues (59,667) | 37 leagues, two seasons apart (45,975) |
|---|---|---|---|
| version 1 (rating, transfer league, goal difference) | 0.489 | 0.503 | 0.431 |
| rating, transfer league, Elo team | 0.537 | 0.537 | 0.464 |
| **version 2** | **0.859** | **0.858** | **0.656** |
| version 2 from one season only (check) | 0.596 | 0.595 | 0.544 |

Version 2 pools two seasons, so a score and the next season's share one
season, which raises the year to year number by itself. The two checks
without that overlap still put version 2 well ahead: one season only 0.595
against 0.503, and two seasons apart (no shared season) 0.656 against
0.431. All gaps have 95% bootstrap intervals well clear of zero.

**Backtest** (all 37 leagues, 2 seasons ahead, each version's ability
columns in the same LightGBM model). Baseline: the success rate of training
season players with a similar value rank and age.

| Target | Model | AUC | Average precision | Top 50 precision |
|---|---|---|---|---|
| elite (9,579 rows, 606 hits) | baseline | 0.874 | 0.295 | 0.367 |
| | version 1 | 0.880 | 0.324 | 0.395 |
| | rating, Elo team | 0.879 | 0.324 | 0.393 |
| | **version 2** | **0.881** | **0.334** | **0.404** |
| step up (82,029 rows, 1,490 hits) | baseline | 0.912 | 0.156 | 0.231 |
| | version 1 | 0.925 | 0.182 | 0.289 |
| | rating, Elo team | 0.925 | 0.181 | 0.284 |
| | **version 2** | **0.927** | **0.186** | **0.295** |

The ability score adds little to these models on top of the value, age and
per 90 columns they already have: the baseline alone gets most of the AUC.

**Performance target, not like for like.** Its label is built from each
version's own ability percentile, so each version predicts a different
target: version 1 AUC 0.803 (2,323 risers), Elo team mix 0.809 (2,284),
version 2 0.809 (1,383). A steadier score has fewer 20 point rises, so
version 2's target is rarer and its average precision lower (0.105 against
0.172).

**Look ahead check.** Version 2 picks its measures from step up outcomes of
all seasons. Repeated with measures from seasons up to 2017 only (labels
known before the 2020 test season), test seasons 2020 on: elite AUC 0.880
and step up 0.941, against 0.879 and 0.942 with all seasons and 0.880 and
0.939 for version 1. The choice of measures does not leak.

**Decision (owner's rule):** version 2 replaces version 1 if it is more
stable and its elite and step up AUC are within 0.005 of version 1 or
better. It is (0.858 against 0.503; AUC +0.001 and +0.002), so
`scout scouting` now uses `--strength v2` by default; `transfer` (version
1), `elo` and `transfer_elo` stay selectable.

## Fall back risk and the candidate tables (decisions 1, 2)

Built on 19 October 2026 on all 37 leagues with ability score version 2
(`src/scout/candidates.py`, `reports/candidates.md`, written by
`scout scouting`). No earlier code defined a fall back, so this is the
definition: an elite candidate (regular, value rank #101 to #1,000) whose
value rank at the end of the 2 seasons is **at least twice as bad** as now
(#300 becomes #600 or worse), the mirror of the step up (3 times better).
Same pool and label windows as the elite target. The flags need per match
data, so the tables start in 2015: 7,842 candidates in 2015 to 2023, of
whom 37.7% fell back. A missing raw number raises no flag (injury history
is known for 68% of them).

| Flag | Raised for | Fall back with | without | Ratio |
|---|---|---|---|---|
| under 47% of team minutes | 30.5% | 42.2% | 35.7% | 1.18 |
| ability below the 55th percentile | 17.0% | 37.5% | 37.7% | 1.00 |
| 5 or more matches missed injured | 21.9% | 48.3% | 34.7% | 1.39 |
| value lower than a year before | 32.4% | 53.6% | 30.0% | 1.79 |

With a base rate near 38% no flag can double the risk the way the earlier
17 league notes said (a doubling would need about 33% without and 65%
with). **The ability flag under version 2 carries no signal at any
percentile:** ratios 1.16 below the 20th (2% of players), 1.06 below the
40th, 1.00 below the 55th, 0.98 below the 70th. So there is no percentile
to recalibrate to; the threshold stays at 55 for the owner to decide, and
the report also shows the count table without it. Two reasons: the market
already prices ability into the rank (a #300 player with a low score is
valued for something else, often age), and version 2 is the steadier
score, so a low one is less often a passing dip. With the looser
definition "out of the top 1,000 two seasons later" the ability flag does
raise the risk (x1.3 at every threshold from the 20th to the 70th), but
that definition punishes a #950 for a small slide.

Fall back rate by flag count, pooled (per season in the report; every
season shows the same order):

| Flags | 0 | 1 | 2 | 3 or more |
|---|---|---|---|---|
| four flags | 25% (2,909) | 40% (2,659) | 50% (1,566) | 55% (708) |
| without ability | 30% (3,522) | 40% (2,481) | 50% (1,353) | 55% (486) |

Elite rate by quintile of the elite model's out of sample score (within
each season, test seasons 2013 to 2023): quintile 1 0%, 2 and 3 under 5%,
4 5%, 5 25% (about 1,900 candidates each). Per season the top quintile
ranges from 20% to 30%.

**Check only, 3 years (not shipped):** 6,919 candidates, 53.2% fall back.
Ratios: minutes 1.07, ability 1.01, injury 1.24, value falling 1.51. No
flag doubles the risk; each one is weaker than at 2 years. Flag counts 0,
1, 2, 3 or more: 45%, 55%, 65%, 70%.

`candidate_profile` returns, for one candidate, the elite chance (quintile,
rounded rate, players behind it), the fall back risk (flag count, rounded
rate, players) and each flag with its raw number; buckets under 30 players
return no rate. `scouting.parquet` carries the flattened profile for the
newest season's candidates.

## Next

1. The clean check above: tune on earlier seasons, score on later ones.
2. Error analysis on the saved predictions: which players the models miss,
   and what their seasons looked like before a big jump.
