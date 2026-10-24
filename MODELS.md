# Models and backtest

The models in use since 19 October 2026 (decision 5, last section): fair value
(`scout backtest`), big move (`scout moves`), elite, step up and riser
(`scout scouting`). Their outputs feed the two user facing tables built by
`scout lists`. **The value growth model is retired**: the growth sections
below are kept as the record of why (it barely beat its age and position
baseline), but the code, its commands and its tests are gone.

`uv run scout backtest` runs the fair value backtest locally, with no AWS
calls. Results: `reports/backtest_fair_value.md` (overall, by league, by
tier, by season) and `reports/backtest_summary.json`; every out of time
prediction goes to `data/processed/models/backtest_fair_value.parquet`.

## Targets

All on a log scale, so an error of 0.10 means about 10%, for a 1m player and a
100m player alike. Only regular players (450+ minutes) are modelled. Row
counts are from the first two league build.

| Target | Label | Rows | May use current value? |
|---|---|---|---|
| `fair_value` | log of the market value at the feature date | 11,360 | no: it is the target |
| `growth_24m` (retired) | log of value 24 months later divided by value now | 8,414 | yes |
| `growth_36m` (retired) | the same, 36 months later | 7,233 | yes |

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
| `no_change` | growth only (retired with it): the value stays where it is |
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

## Automatic model tuning (growth 24m, retired)

`scout train` and `scout tune` now take `--target fair_value` only.

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

As first built; the pool, the flags and the buckets changed after the
owner's review (next to last section, "Fixes after the owner's review").

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

(`candidate_profile`, removed in the simplification, returned) for one candidate, the elite chance (quintile,
rounded rate, players behind it), the fall back risk (flag count, rounded
rate, players) and each flag with its raw number; buckets under 30 players
return no rate. `scouting.parquet` carries the flattened profile for the
newest season's candidates.

## Bargains (decision 4)

Score: market value divided by the fair value model's out of time
prediction, ranked as a percentile within position group and season among
eligible players (ability version 2 at the 70th percentile or better, half of
team minutes or more). Two lists: prime (29 and under) and experienced (30
and over). Contract expiry from `player_profiles.parquet`, shown only when it
ends after the Transfermarkt release date (5 September 2026); never an input.
Warning flags (injured, short record, value falling, lost his place, on loan,
stale value, age 33+) are the module's own until the fall back flags exist.

Backtest, seasons 2015 to 2023: each season's top 50 of each list against
the other eligible players of the same season, ability band and age band.

| list | outcome | players | rate | matched | z |
|---|---|---:|---:|---:|---:|
| prime | value up after 24 months | 390 | 0.618 | 0.431 | 8.1 |
| prime | club with higher Elo within 2 seasons | 450 | 0.398 | 0.415 | -0.7 |
| experienced | value up after 24 months | 360 | 0.119 | 0.044 | 7.1 |
| experienced | club with higher Elo within 2 seasons | 450 | 0.182 | 0.175 | 0.4 |

The market does catch up on the cheap players (part of that is a low price
correcting itself, so it overstates a buyer's gain), but clubs did not sign
them up a level more often than their peers. Per season tables in
`reports/candidates.md` (until 19 October `reports/bargains.md`).

## Performance risers (decision 3)

As first built; a level floor was added after the owner's review (next
section).

Built on 19 October 2026 on all 37 leagues. The old performance target
asked for both of the next two seasons at the 70th percentile of the two
season score and their average 20 above now; measured against one season,
it found dips (Bernardo Silva). The new riser is measured on the **one
season** version 2 score (not pooled): the next season at least 20
percentiles above the best of this season and the last, and the season
after still at least that high, for players with at least half of team
minutes this season (`riser_target`). A season not rated as a regular
breaks the rise. The riser model gets the one season score and the base
as extra columns. Backtest season by season, 2 seasons ahead:

| Season | old: players | base rate | top 50 | top 50 baseline | new: players | base rate | top 50 | top 50 baseline |
|---|---|---|---|---|---|---|---|---|
| 2018 | 4,632 | 0.032 | 0.16 | 0.14 | 2,515 | 0.043 | 0.28 | 0.08 |
| 2019 | 6,854 | 0.035 | 0.20 | 0.14 | 3,767 | 0.051 | 0.22 | 0.10 |
| 2020 | 7,839 | 0.031 | 0.14 | 0.06 | 4,361 | 0.046 | 0.18 | 0.14 |
| 2021 | 9,057 | 0.026 | 0.14 | 0.06 | 4,729 | 0.041 | 0.20 | 0.04 |
| 2022 | 9,511 | 0.026 | 0.24 | 0.04 | 4,870 | 0.038 | 0.40 | 0.14 |
| 2023 | 9,865 | 0.027 | 0.20 | 0.06 | 5,090 | 0.039 | 0.30 | 0.18 |
| mean | 47,758 | 0.030 | 0.180 | 0.083 | 25,332 | 0.043 | 0.263 | 0.113 |

AUC 0.821 against 0.632 for the value rank and age baseline (old: 0.809
against 0.608). The new list beats its base rate about six times over in
every season, so it is kept and replaces the old performance list; the old
target stays in the backtest table. Caught in their season's top 10
include Bounou (2018, before Sevilla), Elmas and Ndicka (2018), Djed Spence
(2020). **Caveat:** without the old 70th percentile floor, a rise from the
20th to the 40th percentile counts, and the current list leans to such low
starting scores (many goalkeepers and centre backs, whose composites are
the weakest). Adding a level floor is for the owner to decide.

## Fixes after the owner's review (19 October 2026)

Five fixes to decisions 1 to 4, all 37 leagues, ability score version 2.

**1. Fall back pool.** The elite target had no age limit, so the elite pool
(`elite_pool` in `src/scout/scouting.py`) is now value rank #101 to #1,000
**and 23 or under at the feature date**, for the elite chance and the fall
back tables alike. The definition is unchanged (value rank at least twice as
bad two seasons later). Seasons 2015 to 2023:

| | Before (#101 to #1,000, any age) | After (also 23 or under) |
|---|---|---|
| candidates | 7,842 | 2,511 |
| fall back rate | 37.7% | 17.2% |
| under 47% of team minutes | x1.18 (30.5% raised) | x1.40 (35.0%) |
| 5 or more matches missed injured | x1.39 (21.9%) | x1.33 (17.1%) |
| value lower than a year before | x1.79 (32.4%) | x1.35 (15.1%) |
| ability below the 55th percentile | x1.00 (17.0%) | x1.26 (19.9%) |

Young candidates fall back far less often (a rising 20 year old keeps his
price longer than a 28 year old), and value falling, the strongest flag
before, matters less among them. **The ability flag fails the owner's rule**
(it must raise the risk 1.3 times; 1.26), so it left the count and the
ability percentile is a plain column. Below the 50th percentile it would pass
(x1.33, and x1.36 below the 40th), but picking the threshold after seeing the
result is the selection the rule exists to avoid; it is retested every run
in `reports/candidates.md`. At three years (check only) it raises the risk
1.40 times.

Fall back rate by flag count, three flags (four flags before):

| Flags | 0 | 1 | 2 | 3 or more |
|---|---|---|---|---|
| before, four flags | 25% (2,909) | 40% (2,659) | 50% (1,566) | 55% (708) |
| after, three flags | 15% (1,318) | 20% (788) | 25% (314) | 20% (91) |

The count still orders the risk up to two flags; three or more is no worse
than two (91 players).

**The elite model** still learns from every candidate ranked #101 to #1,000;
the pool only decides where it is evaluated and listed. Trained on the 23 and
under rows alone it fell below its baseline, so that variant was not kept.
Out of sample, 2013 to 2023:

| Elite | Rows | Hits | AUC | Average precision | Top 50 precision |
|---|---|---|---|---|---|
| all ages, model (before) | 9,579 | 606 | 0.881 | 0.334 | 0.404 |
| all ages, baseline (before) | | | 0.874 | 0.295 | 0.367 |
| 23 and under, model trained on all | 2,925 | 394 | **0.774** | **0.344** | 0.342 |
| 23 and under, baseline | | | 0.771 | 0.320 | **0.349** |
| 23 and under, model trained on the young only (dropped) | | | 0.762 | 0.323 | 0.316 |

Among young candidates the value rank and age baseline is almost as good as
the model: the model ranks slightly better overall (AUC, average precision)
and slightly worse at the very top. Much of the earlier lead came from
telling young from old players, which the baseline also knows.

**2. Elite chance in deciles.** Ten buckets of the out of sample score within
each season instead of five; the lowest deciles merge until each bucket holds
30 players (`merge_low`). Pooled, every decile holds about 290 players, so
none merged: deciles 1 to 10 give under 5%, under 5%, 5%, 5%, 5%, 10%, 15%,
20%, 25%, 40% (before, quintiles on all ages: 0%, under 5%, under 5%, 5%,
25%).

**3. Riser floor.** A riser now also ends at the 70th percentile of his
position or better on the one season score: both of the next two seasons at
least 20 above the best of this and last season, and at least 70.

| Riser | Risers | Base rate | AUC (baseline) | Top 50 precision | Top 50, baseline |
|---|---|---|---|---|---|
| before, no floor | 1,077 | 0.043 | 0.821 (0.632) | 0.263 | 0.113 |
| after, 70th floor | 432 | 0.017 | 0.799 (0.610) | 0.093 | 0.027 |

Per test season, top 50 precision after the floor (base rate in brackets):
2018 0.14 (0.021), 2019 0.02 (0.019), 2020 0.06 (0.019), 2021 0.14 (0.018),
2022 0.10 (0.016), 2023 0.10 (0.013). The floor makes the target rarer and
harder: the list still beats the base rate about five times over and the
baseline about three times, but not in 2019, where it only matches the base
rate. The current list no longer leans to low starting scores.

**4. One flag set** (`src/scout/flags.py`): the three core flags above, used
by the candidates and the bargains, plus two context tags shown on the
bargains only and never counted: on loan, and stale value (market value
dated more than 180 days before the feature date). The bargains' own flags
went: injured at 8 matches and value down a quarter (replaced by the core
thresholds, 5 matches and any fall), short record, lost his place and age 33
or over. Bargains need half of team minutes, so the minutes flag never fires
on them; playing time there is already a filter.

**5. Experienced bargains: 30 to 33.** Players of 34 and over are no longer
listed or used as matched peers. Pooled backtest, 2015 to 2023:

| List | Outcome | Before: rate / matched | After: rate / matched |
|---|---|---|---|
| prime | value up after 24 months | 0.618 / 0.431 | 0.632 / 0.433 |
| prime | club with higher Elo | 0.398 / 0.415 | 0.398 / 0.418 |
| experienced | value up after 24 months | 0.119 / 0.044 | 0.110 / 0.045 |
| experienced | club with higher Elo | 0.182 / 0.175 | 0.187 / 0.191 |

## Simplification (decision 5, 19 October 2026)

**Two user facing tables**, built by `scout lists` (`src/scout/lists.py`)
after `backtest`, `moves` and `scouting`, written to `data/processed/lists/`
as Parquet and CSV with `dictionary.csv` (one plain sentence per column):

* `candidates`: one row per regular player ranked #101 to #2,000 by value in
  the newest season (1,792 rows, 25 columns): elite chance (elite pool only),
  step up chance, fall back risk (elite pool only), flags with raw numbers,
  rising tag, big move chance, current and fair value, ability percentile;
  sorted by elite chance, then step up chance.
* `bargains`: the prime (1,249) and experienced (353) lists, 21 columns, with
  the shared flags and the on loan and stale value tags.

Every chance is a decile rate from the out of sample backtest scores, as for
the elite chance: step up and big move deciles are taken among past players
ranked #101 to #2,000 (the players the table shows), so a #1,500 is compared
with players like him. Pooled rates, deciles 1 to 10: step up under 5% (four
deciles), 5% (three), 10%, 15%, 20% (base rate 5.9%, 19,765 players); big move
under 5% (three), 5% (four), 10%, 15%, 25% (base rate 7.1%, 16,647 players,
blank for players already at a top 20 club). Fall back risk is blank outside
the elite pool, where it was never measured; the flags are shown for everyone.
Rising: the riser model's top 50 of the season (11 of them are in the table's
rank range).

**One readable report**, `reports/candidates.md`: both tables' top 50 and
every backtest number next to its baseline. `reports/scouting.md` keeps the
model detail (league strength, composite weights, traits) without lists;
`reports/moves.md` keeps the big move detail; `reports/bargains.md` is no
longer written.

| Model | Measure | Model | Baseline | Base rate |
|---|---|---|---|---|
| fair value (111,348 rows) | error, log units | 0.395 | 0.780 | |
| | rank correlation | 0.908 | 0.641 | |
| big move (112,531) | AUC | 0.928 | 0.803 | 0.014 |
| | top 50 precision | 0.385 | 0.133 | |
| elite, all candidates (9,579) | AUC | 0.881 | 0.874 | 0.063 |
| elite, elite pool (2,925) | AUC | 0.774 | 0.771 | 0.135 |
| | top 50 precision | 0.342 | 0.349 | |
| step up (82,029) | AUC | 0.927 | 0.912 | 0.018 |
| | top 50 precision | 0.295 | 0.231 | |
| riser (25,332) | AUC | 0.799 | 0.610 | 0.017 |
| | top 50 precision | 0.093 | 0.027 | |

**Retired:** the value growth model (`growth_24m`, `growth_36m`, the
`no_change` model, their commands and tests); nothing used its predictions,
so no replacement was needed, and the growth labels stay for the bargains
backtest. The old performance target and its comparison with the riser; the
old bargains (ability top 15%, value outside the top 1,000) and the investment
lists in `reports/scouting.md`; the bargains' own flags; `scout bargains`
(now part of `scout lists`); `candidate_profile`, the quintile tables, the
ability scan and the three year fall back check, whose numbers are recorded
above. The strength variants (`transfer`, `elo`, `transfer_elo`) stay
selectable; `v2` is the one default path.

## Explanations, two fixes and the first registered forecast (prompt 14, 21 October 2026)

**Explanations** (`src/scout/explain.py`). Both tables carry `explanation`:
the three stats where the player stands furthest from the regulars of his
position group and age band that season (37 leagues pooled), each as a plain
sentence with his number and where it ranks, for example "Takes 13% of his
team's shots, top 1% of wingers aged under 21." Rule based templates for 19
stats (outfield, goalkeeper only, or all); a stat counts when it is in the
top or bottom 10%, a band with fewer than 30 peers falls back to the position
group, and "top 4%" means 4% of peers are at least as good, so a value many
share (no yellow card) never reads as the top. Flags and explanation read
one peer table, and a flag column is told only on the side its flag says
(low minutes only as a weakness when the flag is raised, never as one when
it is not), so they cannot contradict each other. 8.6% of candidates get the
neutral sentence ("No stat stands out against other centre backs aged 24 to
26 this season (15 compared)"); none is empty.

**Fall back by flag count: 2 or more is one bucket.** The 3+ bucket was
small and dipped below 2 (20% against 25%). Now 0, 1, 2 or more: 15%
(1,318 players), 20% (788), 25% (405) over 2015 to 2023.

**Rising rows.** The riser model's current top 50 are rows of the candidates
table even when valued below #2,000: **38 rows added** (36 ranked below
#2,000, 2 without a market value), 1,830 rows in all. 11 risers were already
in the table and 1 is in the top 100, which the table does not list. The
added rows have every chance blank (never measured there), the rising tag
set, and flags and explanation like any other row.

**The young pool (limit).** Among players 23 and under, the elite model
barely beats value rank plus age: AUC 0.774 against 0.771 for the
`rank_rate` baseline in the prompt 12 run, and its top 50 is slightly worse
(0.342 against 0.349). The market's own price already holds most of what the
model knows about a young player's chance of becoming elite. The product's
value is therefore not prediction beyond the market but calibrated odds (a
historical rate per bucket, with its count), risk flags with their raw
numbers, and the same treatment across 37 leagues, where nobody prices every
player by hand.

**Registered forecast and its scoring** (`src/scout/forecast.py`,
`scout forecast`). `forecasts/2026-10-21/` holds the two tables of 21
October 2026 (season 2025, feature dates 30 June 2026, 1 January 2026 for
the eight calendar year leagues; Transfermarkt valuations end 12 June 2026)
with the dictionary, `forecast.json` and a README of what each column
promises. `scout forecast score FOLDER --as-of DATE` scores a folder from
later seasons' value ranks and the transfers, using nothing dated after
`--as-of`. `scout forecast history` rebuilt the tables of every season 2015
to 2023 in the same format (backtest scores, chance tables only from seasons
whose two year outcome was known by then) and scored each two seasons on.
Pooled:

| promise | bucket shown | players | happened |
|---|---|---:|---:|
| elite chance | under 5% / 5% / 10% / 15% | 322 / 803 / 249 / 378 | 3.7% / 3.5% / 7.6% / 13.0% |
| | 20% / 25% / 30% / 40% | 167 / 246 / 76 / 141 | 15.0% / 23.6% / 23.7% / 30.5% |
| | base rate | 2,440 | 11.3% |
| fall back risk | 15% / 20% / 25% / 30% | 854 / 747 / 260 / 64 known | 13.8% / 19.8% / 22.7% / 23.4% |
| | base rate | 1,925 known | 17.7% |
| step up chance | under 5% / 5% / 10% / 15% / 20% | 6,575 / 4,378 / 2,799 / 1,057 / 878 known | 0.3% / 3.5% / 11.2% / 14.8% / 22.9% |
| | base rate | 15,687 known | 5.4% |
| big move chance | under 5% / 5% / 10% / 15% / 25% / 30% | 4,861 / 4,126 / 1,774 / 1,177 / 1,181 / 150 | 1.4% / 4.9% / 9.5% / 15.0% / 26.6% / 24.0% |
| | base rate | 13,269 | 7.3% |
| bargains, value rose | prime top 50 against the list | 394 / 7,947 known | 67.0% / 47.6% |
| | experienced top 50 against the list | 355 / 1,941 known | 14.4% / 7.1% |

Step up, big move and fall back are calibrated within a few points. The
elite chance is about right up to 25% but **too high at the top**: the 40%
bucket made it 30.5% of the time (the 35% and 45% buckets, 28 and 30
players, 42.9% and 36.7%). The current table's top decile also says 40%, so
read it as about 30%. The scorer's outcomes (later seasons' feature rows)
agree with the stored model labels (the valuation two years on) on 98.4%
(elite), 98.9% (step up), 98.2% (fall back), 99.9% (big move) and 96.7%
(value rose) of rows. Details per season in `reports/forecast_history.md`.

## Shown chances are realized rates of past forecasts (prompt 14b, 21 October 2026)

The prompt 14 scoring showed the elite chance too optimistic at the top
(shown 40%, happened 30.5%). Every shown chance was an in sample rate: the
backtest rows of every test season pooled per bucket. Since prompt 14b each
chance (elite by decile, step up and big move by decile, fall back by flag
count) is the **realized rate of the rebuilt past forecasts**: of the players
that `scout forecast history` put in the bucket, with the models and buckets
of their season, the share whose outcome came true two seasons on
(`realized_tables` in `src/scout/candidates.py`). The 30 player floor now
counts players with a known outcome; rounding to 5% stays; the count shown is
that number. A forecast for season S uses forecast seasons up to S minus 2
only, so the 2025 table uses 2015 to 2023 (all scored by the 30 June 2026
feature date). `scout forecast history` runs twice: the first pass (in sample
rates, `history/in_sample/`) fixes each row's bucket and outcome in
`history/outcomes.parquet`, which the buckets do not depend on; the second
shows the realized rates. `scout lists` now runs after `forecast history`.

Old against new, every bucket of the current table (n: players behind the
rate; old counts backtest rows, new counts forecast players with a known
outcome):

| promise | bucket | old | new | new, unrounded |
|---|---|---|---|---:|
| elite | decile 1 / 2 | under 5% (287 / 294) | under 5% (240 / 244) | 1.7% / 2.5% |
| | decile 3 / 4 / 5 | 5% (292 / 293 / 295) | 5% (243 / 246 / 244) | 3.3% / 4.5% / 5.3% |
| | decile 6 | 10% (290) | 10% (242) | 9.5% |
| | decile 7 | 15% (292) | 15% (245) | 13.9% |
| | decile 8 | 20% (293) | **15%** (244) | 15.6% |
| | decile 9 | 25% (293) | **20%** (243) | 21.4% |
| | decile 10 | 40% (296) | **35%** (249) | 34.5% |
| step up | deciles 1 to 4 | under 5% (about 1,975 each) | under 5% (1,543 to 1,570) | 0.0% to 0.6% |
| | decile 5 | 5% (1,977) | **under 5%** (1,573) | 1.7% |
| | decile 6 / 7 | 5% (1,976 / 1,975) | 5% (1,573 / 1,575) | 3.1% / 5.7% |
| | decile 8 | 10% (1,977) | 10% (1,576) | 8.6% |
| | decile 9 | 15% (1,975) | **10%** (1,575) | 12.4% |
| | decile 10 | 20% (1,983) | 20% (1,583) | 21.4% |
| big move | deciles 1 to 3 | under 5% (about 1,665 each) | under 5% (about 1,325 each) | 0.5% / 1.1% / 1.3% |
| | deciles 4 to 7 | 5% (about 1,665 each) | 5% (about 1,327 each) | 2.8% / 3.8% / 4.6% / 7.4% |
| | decile 8 / 9 / 10 | 10% / 15% / 25% (1,665 / 1,665 / 1,670) | 10% / 15% / 25% (1,328 / 1,327 / 1,331) | 10.2% / 14.8% / 26.3% |
| fall back | 0 / 1 / 2 or more flags | 15% / 20% / 25% (1,318 / 788 / 405) | 15% / 20% / 25% (1,298 / 720 / 384) | 15.3% / 20.3% / 24.7% |

Every realized rate rises from bucket to bucket; the report
(`reports/candidates.md`) checks it and would name any bucket out of order
rather than smooth it. 99 elite pool players show a lower elite chance, 359
candidates a lower step up chance; no player changed bucket, order or flags,
and the bargains table is unchanged.

**Out of sample check** (`reports/forecast_history.md`). The rebuilt
forecasts of 2017 to 2023 now show realized rates from forecast seasons two
or more earlier (2015 and 2016 have none before them). Shown against what
happened, pooled, next to the in sample rate on the same seasons:

| promise | realized rate shown: happened | in sample shown: happened |
|---|---|---|
| elite | 15%: 14.3%, 20%: 20.1%, 25%: 33.3% (84), 35%: 29.8% (57), 40%: 32.5% (114) | 15%: 12.1%, 20%: 15.0%, 25%: 22.1%, 40%: 30.5% (141) |
| fall back | 15%: 13.9%, 20%: 18.1%, 25%: 18.9%, 30%: 24.7%, 35%: 23.4% | 15%: 13.8%, 20%: 19.8%, 25%: 22.7%, 30%: 23.4% |
| step up | 5%: 5.9%, 10%: 10.1%, 15%: 13.5%, 20%: 22.5% | 5%: 3.5%, 10%: 10.4%, 15%: 14.8%, 20%: 22.9% |
| big move | 5%: 4.3%, 10%: 9.6%, 15%: 14.1%, 25%: 26.5% | 5%: 4.5%, 10%: 9.0%, 15%: 15.5%, 25%: 26.5% |

The realized rates fix the middle of the elite chance (20% now happens
20.1%, was 15.0%) and narrow the top gap (40% shown, 32.5% happened, against
30.5%), but the top still runs high out of sample by about 6 to 8 points.
Fall back gets worse at the top (35% shown, 23.4% happened): the 30% and
35% shown buckets come from 2017 to 2019, whose realized tables rested on
one to three seasons with higher fall back rates. Early tables are thin;
the current one rests on nine seasons. Its top elite decile (34.5% pooled)
made it 37.5%, 42.3% and 42.9% in 2015 to 2017 but 27.6% to 32.1% in each
season 2020 to 2023 (about 28 players a season): the pooled rate leans on
the early seasons, so the 35% shown is likely still a few points high.
Treat the top elite chance as a ceiling rather than a midpoint; weighting
recent seasons would be the owner's call.

The registered forecast `forecasts/2026-10-21/` was revised in place with
the same date (`scout forecast revise`): `forecast.json` and its README
record the revision, the reason and every bucket's old and new value; the
original is commit e012805, and the revised one carries the git tag
`forecast-2026-10-21`.

## A rolling window of five scored seasons (prompt 14c, 21 October 2026)

The prompt 14b check showed the realized rates pooled from 2015 on still
leaning on 2015 to 2017: the top elite decile made it 27.6% to 32.1% in each
season 2020 to 2023 against 35% shown, and the top fall back bucket showed
35% against 23.4%. The owner's decision: every shown chance (elite by
decile, step up and big move by decile, fall back by flag count) pools only
the **last five scored seasons**, so a forecast for season S reads seasons
S minus 6 to S minus 2 and the 2025 table reads 2019 to 2023
(`WINDOW` and `realized_tables` in `src/scout/candidates.py`). The 30
player floor on known outcomes, the rounding to 5% and the count stay. A
bucket under the floor merges with its neighbour (deciles upward from the
bottom, a short top bucket into the one below; flag counts the same way, as
"1 or more"), and `reports/candidates.md` names every merge. In the 2025
table no bucket fell under the floor.

Old (every scored season, 2015 to 2023) against new (2019 to 2023), every
bucket of the 2025 table, n in brackets:

| promise | bucket | old | new | new, unrounded |
|---|---|---|---|---:|
| elite | decile 1 | under 5% (240) | **5%** (137) | 2.9% |
| | decile 2 | under 5% (244) | under 5% (140) | 0.7% |
| | decile 3 / 4 / 5 | 5% (243 / 246 / 244) | 5% (138 / 141 / 139) | 2.9% / 4.3% / 4.3% |
| | decile 6 | 10% (242) | 10% (139) | 8.6% |
| | decile 7 / 8 | 15% (245 / 244) | 15% (140 / 139) | 13.6% / 16.5% |
| | decile 9 | 20% (243) | 20% (139) | 21.6% |
| | decile 10 | 35% (249) | **30%** (142) | 31.7% |
| step up | deciles 1 to 5 | under 5% (1,543 to 1,573) | under 5% (865 to 876) | 0.0% to 1.3% |
| | decile 6 / 7 | 5% (1,573 / 1,575) | 5% (877 / 876) | 3.5% / 6.5% |
| | decile 8 / 9 | 10% (1,576 / 1,575) | 10% (877 / 875) | 9.9% / 12.0% |
| | decile 10 | 20% (1,583) | **25%** (878) | 22.9% |
| big move | deciles 1 to 3 | under 5% (about 1,325 each) | under 5% (about 736 each) | 0.7% / 0.8% / 1.2% |
| | deciles 4 to 6 | 5% (about 1,327 each) | 5% (about 736 each) | 2.7% / 4.1% / 5.2% |
| | decile 7 | 5% (1,326) | **10%** (736) | 7.6% |
| | decile 8 / 9 / 10 | 10% / 15% / 25% (1,328 / 1,327 / 1,331) | 10% / 15% / 25% (737 / 736 / 739) | 10.9% / 16.2% / 26.4% |
| fall back | 0 / 1 / 2 or more flags | 15% / 20% / 25% (1,298 / 720 / 384) | 15% / 20% / 25% (691 / 418 / 260) | 13.6% / 21.1% / 22.7% |

Elite decile 1 (2.9%, 4 of 137) now sits above decile 2 (0.7%, 1 of 140):
shown as it is, not smoothed. The candidates used to be sorted by the
unrounded rate, which would have lifted decile 1 players above deciles 2
and 3; they are now sorted by elite decile, then step up decile, which
gives the prompt 14b order exactly while the rates rise with the decile
(checked: revision 1 rebuilds byte for byte under either sort). 33 elite
pool players show a lower elite chance and 32 a higher one, 180 a higher
step up chance, 145 a higher big move chance; no fall back risk changed.
Players, order, deciles, flags, explanations, values and the bargains table
are identical to revision 1 (checked column by column).

**Out of sample check** (`reports/forecast_history.md`, section "Old
against new"). `scout forecast history` now runs a third pass with the
prompt 14b rule (`history/all_seasons/<season>/`) next to the five season
one (`history/<season>/`), on the same players, buckets and outcomes. A
test season up to 2021 has at most five scored seasons before it, so both
rules show the same there; they differ only in 2022 (2016 to 2020 against
2015 to 2020) and 2023 (2017 to 2021 against 2015 to 2021), on 140 of 1,956
elite rows and 148 of 10,351 big move rows, and on no step up or fall back
row. Shown (averaged over the bucket's players) against happened, 2017 to
2023:

| promise | bucket | happened | shown, every season | shown, last five |
|---|---|---:|---:|---:|
| elite | decile 1 / 2 / 3 | 2.1% / 1.0% / 2.6% | 0.7% / 5.7% / 4.3% | 1.4% / 5.0% / 3.6% |
| | decile 4 / 5 / 6 | 4.1% / 4.1% / 9.3% | 4.3% / 7.2% / 10.0% | same |
| | decile 7 / 8 | 13.3% / 14.8% | 15.0% / 15.7% | same |
| | decile 9 | 23.1% | 20.7% | 22.1% |
| | **decile 10** | **33.2%** (199) | **36.5%** | **36.5%** |
| fall back | 0 / 1 flags | 14.7% / 20.0% | 17.2% / 20.0% | same |
| | **2 or more** | **22.8%** (324) | **29.6%** | **29.6%** |
| step up | deciles 1 to 10 | 0.0% to 22.3% | within 3.0 points | same |
| big move | deciles 1 to 10 | 0.5% to 26.5% | within 2.4 points | within 1.7 points |

Mean absolute gap, weighted by known outcomes: elite 2.0% (every season)
against 1.7% (last five), fall back 2.5% against 2.5%, step up 0.8%
against 0.8%, big move 1.1% against 1.0%.

* **Top elite decile.** Shown 40% in 2020 and 2021 (both rules) and 35% in
  2022 and 2023 under both rules: in those two seasons rounding hid the
  difference. It made it 31.0%, 30.8%, 32.1% and 27.6%. For the 2025 table
  the window lowers it from 34.5% to 31.7%, shown 30%, inside that recent
  range.
* **Top fall back bucket.** The 35% shown bucket (64 known, 23.4% happened)
  and the 30% one (24.7%) are the same under both rules (no fall back row
  differs; the thin early tables of 2017 to 2019 made them); two or more flags made it 19.6%, 23.8%, 12.8% and 25.6% in 2020 to
  2023 against 30%, 30%, 25%, 25% shown. In the 2025 table it stays 25%
  (22.7% unrounded, from 24.7%).

The honest reading: the out of sample test cannot separate the two rules
well, because five seasons of history only exist from 2022 on. What it
shows is that every season from 2020 to 2023 ran below the long pooled
rate at the top, which is the case for the window; the 2025 table is the
first where the window moves the top elite bucket.

`forecasts/2026-10-21/` is revision 2 (`scout forecast revise`): same date
and rows, the reason and every bucket's old and new value in
`forecast.json` and its README; revision 1 is commit 38a8a4b.

## Freshness: club today, season so far, the January window (prompt 16, 24 October 2026)

Nothing here is a model or a model input; it is context next to the
registered forecast, which stays as it is.

**The pull** (`scout current --as-of 2026-10-24`, 2,031 requests): the team
lists of every league's current season and the five club cups (42), the 830
clubs those lists give plus the league clubs of season 2025 and the cup clubs
we hold, each club's squad (830) and transfer list (830), and the season in
progress of all 37 leagues: the fixture list (37), the 4,807 finished
matches in 255 batches of `/fixtures?ids=`, and the injuries (37). Squads,
transfers, fixture lists and injuries are cached per snapshot day (`as_of`
in the cache key, never sent), so a later day asks again and the same day
never does. Two source errors were repaired in the match adapter, each
logged and tested, with the contract still rejecting the unrepaired value:
Chapecoense's players listed under an unknown team id (22722, no name; two
Brasileirão 2026 matches) are given the other side, and team matches whose
detailed stats are zeros throughout (Ukrainian Premier League 2026/27, 106
of 108 team matches; 26 in K League 1 and 2 in the Czech league) keep their
minutes, goals, assists, cards and penalties with every other stat and the
rating missing. Uruguay sends no player stats for 2026.

**Current club** (`current_club.parquet`, 29,042 players): the club whose
squad lists him today; listed by several clubs (82 players), the club of his
latest move, else the club in one of our leagues; in no squad we hold, the
club his latest move after 30 June 2026 went to (3,494 players, marked
`club_source = transfer`). Of the 1,830 candidates, **619 are at another
club than the club at forecast** (1,190 at the same club, 21 unknown); 436
of the 619 have a move dated after 30 June in the new transfer lists, the
rest mostly loans that ended just before it (Sacha Boey: loan from Bayern to
Galatasaray ended on 29 June 2026; he is back in Bayern's squad, 13
Bundesliga minutes so far). Victor Nelsson: on loan at Hellas Verona at the
forecast, back at Galatasaray on 29 June, a free agent move to FC
Nordsjælland on 31 August (450 Danish Superliga minutes, ability 70th
percentile so far). 517 of the 1,602 bargains moved; 925 of the 2,734
players in either table.

**Season so far** (`season_so_far.parquet`, `scout features --current`):
16,146 players with league minutes in the season in progress (2026/27, or
the 2026 calendar year; J1 2026/27). 94.2% of the candidates and 92.2% of
the bargains have minutes (92.5% of the 2,734 players); 47.5% and 60.3%
have a one season ability percentile so far (450 minutes or more). The
ability is the version 2 one season score with the stored fit, nothing
refitted; the Elo team term is the club's rating in the league's newest Elo
season up to this one. Ukraine has no ability so far: under half its team
matches carry detailed stats. The league means of the percentile follow
league strength as they should (Premier League 78, Ukraine blank, K League
1 29). Goals and assists per 90 are blank for goalkeepers and centre backs
and under 270 minutes. The flags today: low minutes on this season's share,
injured on matches missed in the 12 months before the pull, value falling as
at the forecast (no valuation after 12 June 2026). The season 2025 feature
partitions came out byte for byte the same.

**Window forecast** (`forecasts/window-2027-01/`, frozen 24 October 2026):
the big move model's top 50 (pool 12,832) and the step up model's top 50
among players ranked #101 to #2,000 (pool 1,413), each pool without the
players already at one of the 24 top clubs (top 20 by Elo after their
latest league match, top 20 by squad value in 2025). Rule: joins a top 20
club by Elo or value rank between 1 January and 3 February 2027 (a
transfer, loan or free move; a loan return is not a new move).
`scout forecast window-score` scores each list against its pool's rate from
the transfers table and every transfer snapshot; before 3 February 2027 it
says it is interim (0 of 50 and 0 of 50 so far, as it should be). The lists
are refreshed and frozen again on 15 December 2026 (`--refreeze`).

## Next

1. The clean check above: tune on earlier seasons, score on later ones.
2. Error analysis on the saved predictions: which players the models miss,
   and what their seasons looked like before a big jump.
