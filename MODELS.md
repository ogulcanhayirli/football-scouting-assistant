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

## Similar players (decision 10, prompt 18, 24 October 2026)

`scout embed --train` (`src/scout/embeddings.py`, report
`reports/embeddings.md`). Nothing here feeds the lists' models; it answers
"who plays like him".

**Style vector** (the baseline): one per regular player season from 2015
with match stats, 93,474 rows (91,326 after one row per player and season,
the league with most minutes). 18 measures: the 14 version 2 per 90 inputs
and four on pitch shares, the player's share of his team's shots, dribble
attempts, passes and duels divided by his share of team minutes (so playing
time is not read as style). Each is shrunk towards the position group and
season mean by minutes, with a prior per measure and group fitted by
empirical Bayes as version 2 does (passes per 90 needs at most 117
minutes, goals of a centre back 1,877), on one season only: pooling two
seasons would share data between the two seasons the identity test
compares. Then standardised within position group and season and clipped
at 4.

**The level taken out.** Within each position group and season every
measure is regressed on a cubic in both ability percentiles (two season and
one season) and only the residual is kept; the residual is then divided by
its spread at that ability (the squared residual on the same cubic), so
neither the mean nor the spread of a measure says how good the player is.
The mean alone left a correlation of 0.090 (neighbours of the bottom and
top tenth at 45.7 and 55.2); the spread step brought it to 0.031. The check, every player of 2023 to 2025 against his ten nearest of
the same position group and season (341,970 pairs):

| Vectors | Correlation of ability percentiles | Neighbours' mean, bottom tenth | top tenth |
|---|---|---|---|
| level kept | 0.571 | 21.6 | 78.7 |
| level removed (the baseline) | 0.031 | 48.8 | 52.0 |
| learned embedding | 0.037 | 48.1 | 52.0 |

**Learned embedding.** One hidden layer of 128 (ReLU) to a unit vector,
numpy on the CPU, InfoNCE in both directions: the same player in two
consecutive seasons is the positive pair, the other players of the same
position group and season in the batch (up to 256) the negatives. Adam,
30 epochs, about 15 seconds. Settings chosen on validation only (trained to
2021, searched between 2021 and 2022): every one of the nine (32, 48 or 64
dimensions, temperature 0.05, 0.1 or 0.2) scored 20.9% to 21.7% against
14.3% for the baseline; 48 and 0.1 was best. The tested model is trained on
seasons up to 2022 (35,700 pairs) and the stored one refit on 2015 to 2025
(59,670 pairs), in `processed/models/stored/embedding.json` with the
shrinkage priors; `scout embed` (the weekly run) embeds every row with it.

**Identity retrieval, 2023 to 2025** (the test that decides): for each
player with regular seasons in two consecutive test seasons, one season is
hidden and searched with the other among every player of that position group
and season, both ways; a hit is his other season among the five nearest.

| Position group | Searches | Random | Baseline (cosine) | Learned |
|---|---|---|---|---|
| attacking mid | 2,486 | 0.58% | 21.6% | 29.7% |
| central mid | 3,770 | 0.39% | 17.5% | 24.8% |
| centre back | 6,580 | 0.22% | 7.1% | 11.1% |
| centre forward | 4,022 | 0.35% | 13.3% | 22.6% |
| defensive mid | 3,110 | 0.48% | 19.5% | 27.7% |
| full back | 5,416 | 0.27% | 12.1% | 19.5% |
| goalkeeper | 2,322 | 0.56% | 5.0% | 5.7% |
| winger | 5,310 | 0.27% | 14.7% | 22.5% |
| **overall** | 33,016 | 0.35% | 13.2% | 19.9% |

**Adopted:** the learned model beats the baseline overall and in every
position group. Goalkeepers barely: their measures (saves, passes, duels)
describe little style, as with the ability score.

**Replacement check** (weak labels): a club sells a player (a paid
transfer) and signs another of the same position group within 60 days
before or after (any arrival but a loan return), each described by his
latest season before his move (at most 400 days old); transfers of the
clubs of the original 17 leagues. Is the signing among the ten nearest of
the departed player, in the signing's season and position group?

| Seasons of the signing | Model | Pairs | Top 10 | Random | Median similarity percentile |
|---|---|---|---|---|---|
| 2023 to 2025 | baseline | 3,621 | 1.02% | 0.65% | 55.0 |
| 2023 to 2025 | learned | 3,621 | 1.52% | 0.65% | 55.5 |
| 2015 to 2025 | baseline | 7,275 | 1.29% | 0.85% | 56.1 |
| 2015 to 2025 | learned | 7,275 | 1.73% | 0.85% | 57.2 |

Clubs replace a player with one of a similar style only a little more often
than chance (2.3 times random with the learned model on the test seasons;
the median signing sits at the 55th percentile of similarity, 50 at random).
Signings follow level, price and need more than the departed player's
profile, so this check is weak support, not a target.

**Search.** No vector database: the 93,474 vectors of 48 numbers (233,272
would fit as easily) are held in memory, and a numpy search over all of them
answers in 0.5 ms, a filtered "similar now" query in 7 ms. Queries
(`scout similar`, and the page's Similar players tab and player panel):
"similar now" (today's players, the newest season, of the same position
group, filters on age, league and ability percentile) and "looks like X at
Y" (X's season at age Y, the age at the feature date, against today's
players aged Y minus 1 to Y plus 1). Examples, top five:

* Rodri at 21 (Villarreal 2016/17): Cauan Barros, J. Hamza, F. Bernal,
  G. Yah, V. Ríos.
* Lamine Yamal now: M. Olise, T. Stavitski, L. Messi, T. Ali, I. Ferrah.
* Kenan Yıldız now: K. Yuruki, E. Ceïde, M. Hinestroza, A. Saint-Maximin,
  C. Barrios.
* Jude Bellingham at 18 (Dortmund 2020/21): Pedro Fernández, A. Tia,
  S. Mladenović, B. Mazurek, I. Alemayehu.
* Mike Maignan now: I. Radu, Fábio, M. Di Gregorio, Joel Robles, M. Bleve.

## The measured greats (prompt 18b, 25 October 2026)

`scout statsbomb download | build` and `scout greats` (`src/scout/providers/statsbomb/`,
`src/scout/greats.py`, `src/scout/greats_check.py`; report `reports/greats.md`).
Nothing here feeds the lists' models.

**Data.** StatsBomb open data (github.com/statsbomb/open-data): La Liga
2004/05 to 2020/21 and the Premier League 2003/04, matches, lineups and
events, cached byte for byte under `data/raw/statsbomb` (rule 4): 1,829
files, 2.83 GB (2,825,097,646 bytes), 904 matches, plus the user agreement
and the logo. Before 2015/16 a season holds Barcelona's league matches only,
and in the early seasons only those Messi played (7 in 2004/05, 17 in
2005/06, 26 in 2006/07); the Premier League 2003/04 is Arsenal's 38;
2015/16 is complete (380). The StatsBomb Public Data User Agreement allows
research and published analysis, not commercial use and not passing the
data on (1.2), and asks every published analysis to carry the StatsBomb
logo (1.4): the page shows results computed from the data with the logo,
never the data. StatsBomb also asks users to register at
statsbomb.com/resource-centre (the owner's step).

**The mapping.** `parse_match` turns one match into our per match player
table (`PLAYER_MATCH_STATS` passes on all 30,714 rows; ids are StatsBomb's
own and the table lives in `processed/statsbomb/`, never mixed with API
Football rows). Minutes come from the lineups on a regulation clock, as API
Football counts them. Where no equivalent exists the measure is missing,
never zero. Three definitions were chosen on the comparison below between
principled alternatives, and say so: passes leave out throw ins, duels
include fouls won and committed (r 0.858 against 0.804 without), and
blocks are shot blocks only (0.726 against 0.368 for every block).

| Column | StatsBomb event | What is counted |
|---|---|---|
| `minutes` | Lineups: positions | time between each spell's from and to (blank: the final whistle), regulation clock (first half capped at 45, second at 90, extra time at 105 and 120) |
| `goals` | Shot | outcome Goal, periods 1 to 4; own goals (Own Goal Against) not counted |
| `assists` | Pass | pass.goal_assist |
| `shots_total` | Shot | every outcome, periods 1 to 4 |
| `shots_on` | Shot | outcome Goal, Saved or Saved To Post |
| `passes_total` | Pass | every pass but throw ins (pass.type Throw-in) |
| `passes_key` | Pass | pass.shot_assist or pass.goal_assist |
| `passes_accurate` | Pass | no pass.outcome (complete), throw ins left out |
| `tackles_total` | Duel | duel.type Tackle, every outcome |
| `blocks` | Block | blocks of a shot (the Block's related event is a Shot) |
| `interceptions` | Interception | every outcome |
| `duels_total` | Duel, 50/50, Dribble, Dribbled Past, aerial_won, Foul Won, Foul Committed | tackles (Duel Tackle), aerials lost (Duel Aerial Lost), aerials won (aerial_won on a Pass, Shot, Clearance or Miscontrol), 50/50s, dribbles, times dribbled past, fouls won and committed |
| `duels_won` | Duel, 50/50, Dribble, aerial_won, Foul Won | tackles with outcome Won or Success (In Play, Out), aerials won, 50/50s Won or Success To Team, dribbles Complete, fouls won |
| `dribbles_attempts` | Dribble | every outcome |
| `dribbles_success` | Dribble | outcome Complete |
| `dribbled_past` | Dribbled Past | every event |
| `fouls_drawn` | Foul Won | every event |
| `fouls_committed` | Foul Committed | every event |
| `cards_yellow` | Lineups: cards | Yellow Card |
| `cards_red` | Lineups: cards | Red Card or Second Yellow |
| `saves` | Goal Keeper | goalkeeper.type Shot Saved, Shot Saved to Post, Penalty Saved (to Post), Save |
| `goals_conceded` | Goal Keeper | goalkeeper.type Goal Conceded or Penalty Conceded; goalkeepers only |
| `penalty_won` | Foul Won | foul_won.penalty |
| `penalty_committed` | Foul Committed | foul_committed.penalty |
| `penalty_scored` | Shot | shot.type Penalty, outcome Goal, periods 1 to 4 |
| `penalty_missed` | Shot | shot.type Penalty, any other outcome |
| `penalty_saved` | Goal Keeper | goalkeeper.type Penalty Saved (to Post) |
| `rating` | none | no equivalent: missing |
| `offsides` | none | no equivalent (the player caught offside is rarely recorded): missing |
| `captain` | none | no equivalent: missing |

**Validation, the same player and match.** La Liga 2015/16 to 2020/21 is in
both sources: 551 of 552 StatsBomb matches found in API Football (same day,
score and club names), 15,434 player matches linked (shirt number with a
name check) where both give minutes. API Football's blank counts read as 0
where its league season records them for anyone, as the style inputs read
them. Difference is StatsBomb minus API Football per match.

| Measure | Player matches | Correlation | Mean StatsBomb | Mean API Football | Difference |
|---|---|---|---|---|---|
| `minutes` | 15,434 | 0.989 | 71.00 | 70.83 | +0.18 |
| `goals` | 15,434 | 0.999 | 0.10 | 0.10 | +0.00 |
| `assists` | 15,434 | 0.963 | 0.07 | 0.07 | -0.00 |
| `shots_total` | 15,434 | 0.985 | 0.88 | 0.86 | +0.02 |
| `shots_on` | 15,434 | 0.982 | 0.32 | 0.33 | -0.00 |
| `passes_total` | 15,434 | 0.955 | 34.48 | 26.35 | +8.14 |
| `passes_key` | 15,434 | 0.940 | 0.62 | 0.65 | -0.03 |
| `dribbles_attempts` | 15,434 | 0.831 | 1.36 | 1.24 | +0.13 |
| `dribbles_success` | 15,434 | 0.776 | 0.77 | 0.66 | +0.11 |
| `duels_total` | 15,434 | 0.858 | 8.39 | 7.34 | +1.04 |
| `duels_won` | 15,434 | 0.810 | 4.01 | 3.67 | +0.33 |
| `tackles_total` | 3,971 | 0.625 (below 0.7) | 1.16 | 0.71 | +0.45 |
| `interceptions` | 15,434 | 0.675 (below 0.7) | 1.06 | 1.25 | -0.19 |
| `blocks` | 3,971 | 0.726 | 0.22 | 0.13 | +0.09 |
| `fouls_drawn` | 15,434 | 0.915 | 1.07 | 0.91 | +0.16 |
| `fouls_committed` | 15,434 | 0.905 | 1.12 | 0.95 | +0.17 |
| `dribbled_past` | 15,434 | 0.676 (below 0.7) | 0.84 | 0.66 | +0.18 |
| `saves` | 15,434 | 0.986 | 0.23 | 0.22 | +0.00 |
| `cards_yellow` | 15,434 | 0.968 | 0.18 | 0.19 | -0.00 |

By season, the measures whose agreement moves:

| Season | Passes r | diff | Completed passes r | diff | Tackles r | Interceptions r | Blocks r |
|---|---|---|---|---|---|---|---|
| 2015/16 | 0.948 | +9.2 | 0.983 | +1.1 |  | 0.675 |  |
| 2016/17 | 0.963 | +7.4 | 0.981 | +1.0 |  | 0.667 |  |
| 2017/18 | 0.963 | +7.1 | 0.979 | +0.6 | 0.320 | 0.656 | 0.433 |
| 2018/19 | 0.974 | +7.1 | 0.988 | +0.9 | 0.764 | 0.691 | 0.910 |
| 2019/20 | 0.984 | +6.6 | 0.996 | +0.8 | 0.791 | 0.672 | 0.870 |
| 2020/21 | 0.994 | +1.3 | 0.990 | -3.9 | 0.822 | 0.623 | 0.675 |

API Football's pass count is the completed passes until 2019/20 (StatsBomb's
completed passes match it within a pass a match, r 0.98) and every pass from
2020/21 (r 0.994, +1.3): a definition change in the source, absorbed by the
season by season standardisation, and worth knowing when reading pass
numbers across seasons in our own data. Tackles and blocks exist in API
Football's La Liga only from 2017/18, and its 2017/18 tackles disagree with
StatsBomb (0.32) while 2018/19 to 2020/21 agree at about 0.8.

**Rule applied: a measure whose match counts correlate below 0.7 is left out
of the StatsBomb vectors** (it reads as the position mean): tackles per 90
(tackles 0.625) and interceptions per 90 (interceptions 0.675). 16 of 18
measures stay. Season level, the same player and season with 450 minutes or
more in both:

| Measure | Player seasons | Correlation |
|---|---|---|
| `goals_per90` | 364 | 0.991 |
| `assists_per90` | 331 | 0.953 |
| `shots_total_per90` | 516 | 0.990 |
| `passes_key_per90` | 516 | 0.982 |
| `passes_total_per90` | 517 | 0.973 |
| `dribbles_success_per90` | 515 | 0.929 |
| `dribble_success_rate` | 484 | 0.661 |
| `duels_won_per90` | 515 | 0.879 |
| `duel_win_rate` | 513 | 0.677 |
| `tackles_total_per90` | 73 | 0.624 |
| `interceptions_per90` | 515 | 0.889 |
| `blocks_per90` | 67 | 0.816 |
| `fouls_drawn_per90` | 516 | 0.976 |
| `saves_per90` | 37 | 0.984 |

Interceptions per 90 agree well by season (0.889) though not by match; the
rule was set on the match level before looking.

**The owner's call (prompt 18c): tackles and interceptions are kept in the
StatsBomb vectors.** Their cross source correlation per match is 0.63
(tackles) and 0.68 (interceptions), under the 0.7 rule, but the identity
test, not the correlation, decided: with them the seasons held for
Barcelona find the same player's API Football season in the top five 28.6%
of the time against 14.3% without (table below). The rule stays in the code
and the report as a flag (`MIN_CORRELATION`, `KEPT` in
`src/scout/greats_check.py`); all 18 measures are in the vectors.

**Cross source identity check.** Each StatsBomb regular season of 2015/16
to 2020/21 linked to API Football, embedded with the stored model through
the peer group pipeline below (the player in his API Football position
group), searched among every API Football main row of that season and
group: is his own API Football season among the five nearest?

| Seasons | Peer group | Searches | Top 5 | Random | Median rank | Median pool |
|---|---|---|---|---|---|---|
| 2015/16, with the exclusions | whole league | 392 | 79.1% | 1.11% | 2 | 487 |
| 2016/17 to 2020/21, with the exclusions | Barcelona and opponents | 98 | 14.3% | 0.62% | 25 | 874 |
| all, with the exclusions | both | 490 | 66.1% | 1.01% | 2 | 487 |
| 2015/16, every measure | whole league | 392 | 76.0% | 1.11% | 2 | 487 |
| 2016/17 to 2020/21, every measure | Barcelona and opponents | 98 | 28.6% | 0.62% | 16 | 874 |
| all, every measure | both | 490 | 66.5% | 1.01% | 2 | 487 |
| 2015/16, every measure, corrected (prompt 18c) | whole league | 392 | 75.8% | 1.11% | 2 | 487 |
| 2016/17 to 2020/21, every measure, corrected | Barcelona and opponents | 98 | 55.1% | 0.62% | 5 | 874 |

Where the peer group is the whole league (2015/16) the StatsBomb season
lands next to the same player's own season 79.1% of the time (median rank
2): the mapping measures the same football. Where the peer group is
Barcelona and its opponents (2016/17 on, as every season before 2015/16 has)
it does so 14.3% of the time (median rank 25, 98 searches), above 0.62% at
random but far from the full league. With every measure kept the thin
seasons do better (28.6%); the rule's two exclusions cost information
there, and the owner kept the measures (above). With the correction below
the thin seasons reach 55.1% (median rank 5). 2015/16 does not change with
it (75.8% against 76.0%; only its pooling with the corrected 2014/15 moves).

**Peers and level (before 2015/16).** Each StatsBomb season is standardised
within position group and season against the players in those matches with
90 minutes or more. **This peer group is Barcelona and its opponents only
(Arsenal and its opponents for 2003/04): thinner, since most opponents are
seen once or twice a season, and biased, since every opponent's numbers are
those of a team playing Barcelona.** A thin peer group would inflate a
regular's standardised values (shrunk low minute peers have little spread),
so the true spread of each measure is estimated from the peers by the
method of moments (squared deviation over 1 + k / minutes, k the stored
prior minutes) and a regular's shrunk deviation is divided by it times the
square root of the mean shrinkage weight our own regulars get, which is what
our standardisation divides by on a full league. The level is removed with
an ability percentile computed inside the same peer group (stored version 2
weights over the shrunk measures, pooled with the previous season by
minutes, no league or team term) and the level curves of our own data (a
cubic per position group over 2015 to 2025), since 30 to 60 peers per group
and season are too few to fit one. **Cross era comparisons are rough: our
league and team adjustment starts in 2010 and is not applied to these
seasons at all.**

What the thin peer group costs, measured on 2015/16 where both exist:
Barcelona's 19 regulars standardised against the whole league (502 players)
and against Barcelona and its opponents only (276). The two vectors of the
same player season have a median cosine of 0.68 (lowest 0.50), and his own
API Football season is in the top five 100% of the time with the whole
league against 15.8% with the thin peer group (median rank 1 against 25).
Uncorrected, the pre 2015 greats were rough sketches.

**The correction (prompt 18c, `PeerCorrection` in `src/scout/greats.py`).**
Learned on La Liga 2015/16, held twice: every match, and Barcelona's matches
only (as every other StatsBomb season is held). Per position group and
measure, two numbers: the shift, where the whole league's mean sits in
spreads of the opponents' (the opponents' mean minus the league's, over
the opponents' true spread; negative means Barcelona's opponents did less
of it), and the ratio of the opponents' true spread to the league's. The
opponents are the reference, not the whole thin group, so the players being
measured never set their own baseline. In a season held for Barcelona the
league's mean and spread are rebuilt from that season's opponents with the
two numbers, every player is standardised against the rebuilt league, and
his ability percentile, which takes the level out, is placed in the 2015/16
league's distribution of the same score instead of ranked among the thin
peers. A measure or group without both numbers keeps the opponents' own.

The opponents of Barcelona won fewer duels (shifts of -1.3 to -2.6 spreads
in every outfield group), drew fewer fouls, and among central midfielders
scored, assisted and shot far less (about -3); spreads are true spreads, so
a measure with very little real variation shows a large shift (a centre
back's share of passes, -13.3). Every pair is in `reports/greats.md` and
`processed/greats/correction.csv`.

In sample (2015/16, where it was learned, so a check and not evidence):
the median cosine between a Barcelona regular's thin and whole league
vectors rises from 0.68 to 0.93, and his own API Football season is in the
top five 94.7% of the time against 21.1% uncorrected (100% with the whole
league; short of exact because the check puts each linked player in his
API Football position group while the correction was learned with
StatsBomb's).

**The test, 2016/17 to 2020/21** (Barcelona's matches only, after the
season it was learned on; every measure kept; the rule, set before the
test: applied when the top five rate rises and the median rank does not):

| Vectors | Searches | Top 5 | Median rank | Better | Worse |
|---|---|---|---|---|---|
| uncorrected | 98 | 28.6% | 16 | | |
| corrected | 98 | 55.1% | 5 | 66 | 27 |

By season, corrected: 63.2% (2016/17), 50.0%, 82.4%, 31.6%, 52.4%
(2020/21). **It helps, so it is applied to every season held for Barcelona
before 2015/16, and the page says what it does.** Arsenal's Premier League
2003/04 stays uncorrected: the correction was learned and tested on
Barcelona's opponents in La Liga only, and the one look at that season
(Henry's, in the archetypes' calibration below) showed the La Liga shifts
misplacing a Premier League forward. Arsenal's opponents' forwards drew
more fouls (2.42 per 90) than the thin group's mean (2.01), where
Barcelona's opponents drew fewer than the league, so the rebuilt league
mean (3.49) put Henry's 1.79 at the bottom of every forward. Henry's season
keeps the thin peer group (Arsenal and its opponents) and its caveat.

**The greats.** 23 greats, 239 seasons, one per age: Messi from 17 to 38
(StatsBomb to 2014/15, then our own data through Paris and Miami), Xavi,
Iniesta, Busquets, Puyol, Piqué, Dani Alves, Henry, Vieira, Bergkamp and
Pirès from StatsBomb (and our own data from 2015/16 for those still
playing), and Ronaldo, Modrić, Ramos, Lewandowski, Kroos, Neymar, Suárez,
De Bruyne, Kanté, Van Dijk, Buffon and Chiellini from our own data (2015
on). Ages are taken in the middle of the season (1 January; 1 July in
calendar year leagues), so Messi at 19 is 2006/07, and today's players are
aged the same way. A season under 450 minutes is kept and marked thin (9 of
239, Messi at 17 among them with 74 minutes): mostly the position mean.

Examples (today's players of that age and the great's position group; the
elite chance is the calibrated one of the candidates table, which exists
only for the elite pool, 23 or under):

With every measure kept and Barcelona's seasons corrected (prompt 18c; the
18b lists before it are in git):

* **Messi at 19** (2006/07, 2,001 minutes): B. Touré (Hoffenheim, 0.63,
  elite chance 20%), J. Bal (ADO Den Haag, 0.55), Y. Diomande (RB Leipzig,
  0.53), Jan Virgili (Mallorca, 0.52, 15%), Abdoulaye Camara (Vejle, 0.52).
  In the elite pool: B. Touré, Jan Virgili, S. El Mala (30%).
* **Xavi at 25** (2005/06, a thin season: 437 minutes in the 17 matches
  StatsBomb holds): L. Johnsen (Malmö, 0.74), Tiago Dantas (Rijeka),
  O. Kökçü (Beşiktaş), S. Mezei (TSC), J. Zapata (Atlético Nacional). At 26
  (2006/07, 2,002 minutes): I. Louza (Watford, 0.59), U. Bislimi (Lugano),
  R. Sedláček (Jablonec), Matheus Bueno (Shimizu), Monchu (Aris).
* **Henry at 26** (Arsenal 2003/04, 3,328 minutes, uncorrected): E. Latte
  Lath (Atlanta, 0.78), J. Ardaiz (Sarmiento), P. Judd (San Jose),
  S. Surridge (Nashville), M. Toye (Sporting Kansas City).

Similar means plays the same way with the level taken out, so the nearest
are rarely stars; the page labels every result "style similarity from box
score stats is coarse; the same player is found in the top five about one
time in five; treat lists as shortlists".

**Garbled names (rule 7).** API Football sends some names as UTF-8 read as
Latin-1 or Windows-1252 ("O. H\u00c3\u00b8jlund") or with HTML entities
("M. O&apos;Riley"). `scout.providers.api_football.names.clean_name`
repairs both in every adapter (a run that does not decode to a letter is
left alone), and the contracts reject a name it would still change. 741
distinct names in 379 of 1,986 processed partitions were affected, far more
than the ten first noticed (most were the `&apos;` of N'Diaye, O'Brien and
D'Alessandro); `scripts/repair_names.py` applied the same repair to them
(a pure function of the name, so the same as parsing the cache again). On
the 2025 tables 9 candidates' and 3 bargains' names changed.

## Archetypes (prompt 18c, 25 October 2026)

`scout archetypes` (`src/scout/archetypes.py`; files under `archetypes/`;
report `reports/archetypes.md`). Nothing here feeds the lists' models.

**What an archetype is.** A player we have no data for, described from the
historical record: one YAML file per player with his position group, era,
`measured: false`, each of the 18 style measures as a percentile against
the regular players of his position group in his own seasons (before the
level is taken out) with a one line reason, and the sources. Eight legends
(`archetypes/legends/`: Cruyff and Ronaldo Nazário as centre forwards,
Gullit and Zidane as attacking midfielders, Pirlo as a defensive
midfielder, Rio Ferdinand and Baresi as centre backs, Maldini as a left
back) and six calibration players (`archetypes/calibration/`). **Every file
was drafted by this Claude Code session on 25 October 2026 from the
historical record and awaits the owner's review**; the session watched no
matches and did not open the listed sources while drafting. The drafts were
committed on their own (f056480) before any calibration season was looked
at.

**Into the level free space.** A percentile becomes a standardised value:
the value at that percentile of his position group's distribution over our
own regular player seasons from 2015 (one row per player and season, level
kept). His ability percentile is the stored version 2 weights over those
values, placed in the same seasons' distribution of that score. The level
is taken out with the curves our own players go through (the cubic per
position group of the measured greats), and the stored model embeds the
result, so an archetype is searched among today's players exactly as a
measured season is. Every result is labelled "archetype from the record,
not measured". The version 2 weights favour what a rising player shows
(centre backs: goals, passes, shots; Open questions), so a defensive legend
drafted with few goals gets a middling ability (Baresi 51, Ferdinand 47)
and the level curves treat him as a middling centre back.

**Calibration.** Six players we can measure, drafted the same way before
their data was looked at: Modrić (Real Madrid 2018/19), Iniesta (Barcelona
2015/16), Van Dijk (Liverpool 2018/19) and Kanté (Leicester 2015/16) in our
own data; Messi at 25 (Barcelona 2012/13, corrected) and Henry at 26
(Arsenal 2003/04, uncorrected) in StatsBomb's. Each drafted percentile is
set against the measured one on the same scale (his standardised value
placed among our regular player seasons of his position group). A trait
without spread in the group (an outfield player's saves) or without data
that season (API Football's La Liga has no tackles or blocks before
2017/18) is not compared. The baseline drafts 50 on every trait.

| Player | Season | Traits | Mean gap | Within 15 | Baseline mean gap | Baseline within 15 |
|---|---|---|---|---|---|---|
| Henry at 26 | 2003/04, StatsBomb | 17 | 20.7 | 47% | 33.2 | 24% |
| Iniesta | 2015/16, API Football | 15 | 18.6 | 33% | 24.0 | 40% |
| Kanté | 2015/16, API Football | 17 | 20.0 | 41% | 32.3 | 12% |
| Messi at 25 | 2012/13, StatsBomb | 17 | 12.2 | 65% | 42.9 | 6% |
| Modrić | 2018/19, API Football | 17 | 14.8 | 59% | 21.4 | 47% |
| Van Dijk | 2018/19, API Football | 15 | 14.9 | 53% | 27.5 | 40% |
| the 4 in our own data | | 64 | 17.1 | 47% | 26.3 | 34% |
| **all 6** | | 98 | 16.9 | 50% | 30.4 | 28% |

**Verdict, by the owner's rule (within 15 points on most traits): the
drafts land within 15 points on 49 of 98 traits, exactly half, not most, so
the archetypes are illustrative only**, and the page says so next to them.
They carry real information (mean gap 17 points against 30 for the
baseline; 50% within 15 against 28%) but are wrong by more than 15 points
on half the traits. Where the record helps and where it misleads: goals,
shots, passes and dribbles per 90, the volumes it reports, landed within
15 points on 20 of 24. The two rates were missed by more than 15 on 9 of
12, the duel win rate mostly drafted too low (Iniesta 60 against 98, Messi
70 against 100) and one dribble success far too high (Kanté 70 against
21). Assists of the deeper players were drafted too low (Kanté 45 against
89, Modrić 70 against 89, Van Dijk 65 against 85). Defensive volumes were
missed both ways (Modrić's tackles 50 against 20, Van Dijk's interceptions
40 against 8, but Henry's interceptions 30 against 92 and blocks 25 against
93 in the thin Arsenal season). The share of the team's passes was missed
both ways (Kanté 45 against 85, Modrić 85 against 50) and the share of its
duels drafted too high for four of five.

Searched with his drafted archetype among every regular of his position
group that season, the real season ranks 1st of 352 (Kanté), 16th of 1,264
(Van Dijk), 17th of 431 (Iniesta) and 121st of 725 (Modrić); top five at
random is 0.4% to 1.4%. The StatsBomb seasons are not searched: a season
held for one club has no pool of other regulars.

**Today's nearest five** (any age, the archetype's position group, the
newest season; similarity on the learned embedding; ability is the
player's own percentile):

* **Cruyff** (centre forward, ability 98 from the drafted traits):
  G. Melkadze (Akhmat, 0.87), N. Mohammed (Halmstad, 0.86), C. Brym (NAC
  Breda, 0.80), Y. Silue (OFK Beograd, 0.77), B. Touré (Annecy, 0.76).
* **Rio Ferdinand** (centre back, 47): Z. Monlouis (Toronto, 0.80),
  Jahmai Simpson-Pusey (Köln, 0.75), E. Mihaj (Atlanta, 0.75), Robert
  Renan (Vasco da Gama, 0.75), J. Quarshie (Southampton, 0.74).
* **Pirlo** (defensive midfielder, 92): Damián Rodríguez (Racing
  Santander, 0.86), M. Castagnetti (Cesena, 0.81), M. Klich (Cracovia,
  0.79), J. Gastien (Clermont, 0.79), Jorginho (Flamengo, 0.76); Kimmich
  is seventh.

The page's Archetypes tab shows each archetype's 18 drafted traits with
their reasons and sources, the review status, the calibration verdict, and
today's ten nearest (age range and elite pool filters). `scout archetypes`
reloads every file after the owner edits it, checks it (every trait, a
percentile from 0 to 100, a reason, a source; a test runs the same check),
rebuilds the vectors, today's neighbours (`processed/archetypes/`), the
calibration and the report; the weekly run calls it before `scout site`, so
an edited file reaches the page at the next run.

## Next

1. The clean check above: tune on earlier seasons, score on later ones.
2. Error analysis on the saved predictions: which players the models miss,
   and what their seasons looked like before a big jump.
