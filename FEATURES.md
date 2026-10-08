# Feature tables

Built by `uv run scout features` from the processed season stats, match stats
and the Transfermarkt snapshot. It spends no API requests. Output under
`data/processed/features/`:

| Table | Path | Rows |
|---|---|---|
| Features | `player_season_features/league=<key>/season=<year>/data.parquet` | 16,264 |
| Labels | `player_season_labels/league=<key>/season=<year>/data.parquet` | 16,264 |
| Links | `player_links.parquet` (API Football id to Transfermarkt id) | 7,189 |

One row per league, season and player with minutes in that league season.
Key: `league_key`, `season`, `source_player_id`. The `season` value lives in
the folder name, not in the file (Athena's rule for partition keys), so read
the tables with `scout.features.pipeline.read_feature_table`, which restores
it. A player who moved within a
league has one row; a player who played in both leagues in one season has a
row in each.

## Querying in Athena

The `ScoutData` stack (`infra/`) registers both tables in the Glue database
`scout`. In Athena the partition keys are `league` and `season`; filter on
them, and a query reads only those folders:

```sql
SELECT player_name, team_name, goals_per90, market_value
FROM player_season_features
WHERE league = 'super_lig' AND season = 2024 AND regular
ORDER BY goals_per90 DESC
LIMIT 10;
```

Join the labels on `league`, `season` and `source_player_id`. From the
command line: `uv run scout athena "<SQL>"`, which runs as the read only
`scout-analyst` role in the `scout` workgroup (1 GB scan limit per query).

## SageMaker Feature Store

The `ScoutFeatureStore` stack holds the same rows in the feature group
`scout-player-season-features` (offline store only, Iceberg):

| Feature Store field | Here |
|---|---|
| record identifier `record_id` | `<league_key>#<source_player_id>`, for example `epl#1100` |
| event time `event_time` | the feature date, ISO 8601 |
| features | `season` plus every column above; booleans as 0 or 1, timestamps as ISO strings |

`uv run scout featurestore ingest` puts every row (writing a row again adds a
version, which point in time queries ignore). `uv run scout featurestore pit
--as-of <date>` asks the offline store for every player as of a date and
compares the answer with our tables. The offline store fills in batches, so
new records show up after about 5 to 15 minutes.

## The point in time rule

Every row describes a player **as of its `feature_date`** and uses nothing
dated after it. The feature date is the later of 30 June and the day after
the season's last league match; that is 27 July 2020 for 2019/20, because of
the COVID restart, and 30 June otherwise.

Two audit columns let the contract check the rule on every build:
`last_match_date` and `value_date` must not be after `feature_date`.
Transfermarkt's contract expiry, international caps and current value are
not used at all: they describe 2026, not the feature date.

## Tiers

| `tier` | Seasons | What exists |
|---|---|---|
| `match` | 2015 to 2025 | everything below, from per match data (D12) |
| `season` | EPL 2010 to 2014, Süper Lig 2011 to 2014 | minutes, appearances, starts, goals, cards and their per 90 rates |

Detailed columns are missing (not zero, not imputed) in the `season` tier
(D10). In the `match` tier, a blank count for a player who played is read as
0 when most players that season have the stat, because the API leaves zeros
blank.

## Feature columns

**Identity and profile**

| Column | Meaning |
|---|---|
| `player_name`, `nationality` | from API Football |
| `tm_player_id`, `link_rule` | Transfermarkt id and the rule that linked it (`name_score` or `surname`); missing when not linked |
| `team_id`, `team_name` | club at the end of the season (most minutes, in the `season` tier) |
| `teams_played_for` | clubs in this league this season |
| `position` | Goalkeeper, Defender, Midfielder or Attacker (API Football) |
| `sub_position`, `foot` | from Transfermarkt, for example Centre-Back; barely change over a career |
| `age` | years at the feature date |
| `height_cm` | API Football, Transfermarkt where missing |
| `regular` | at least 450 minutes |

**Playing time**

| Column | Meaning |
|---|---|
| `minutes`, `appearances`, `starts` | season totals |
| `minutes_share` | minutes divided by 90 times the matches his clubs played while he belonged to them |
| `last_match_date` | his last appearance before the feature date (audit) |

**Output, `match` tier**

| Column | Meaning |
|---|---|
| season totals | `goals`, `assists`, `shots_total`, `shots_on`, `passes_total`, `passes_key`, `passes_accurate`, `tackles_total`, `blocks`, `interceptions`, `duels_total`, `duels_won`, `dribbles_attempts`, `dribbles_success`, `dribbled_past`, `fouls_drawn`, `fouls_committed`, `cards_yellow`, `cards_red`, `saves`, `goals_conceded`, `offsides`, `penalty_won`, `penalty_committed` |
| `<total>_per90` | each total per 90 minutes |
| `rating` | API Football match rating, averaged with minutes as weights |
| `shot_accuracy`, `pass_accuracy`, `duel_win_rate`, `dribble_success_rate` | success shares, 0 to 1 |

**Team and form, `match` tier**

| Column | Meaning |
|---|---|
| `team_matches`, `team_ppg`, `team_gd_per_match` | his final club's league matches, points per game and goal difference per match |
| `form_minutes_share` | share of the minutes in his club's last 10 matches |
| `form_involvement_per90` | goals plus assists per 90 in those matches |
| `form_rating` | mean rating in those matches |

**Market value (Transfermarkt), as of the feature date**

| Column | Meaning |
|---|---|
| `market_value`, `value_date` | latest value on or before the feature date, at most a year old |
| `market_value_1y_before` | the same, one year earlier |
| `value_change_1y` | `market_value` divided by `market_value_1y_before` |
| `market_value_max_to_date` | highest value up to the feature date |

## Label columns

| Column | Meaning |
|---|---|
| `value_now` | value at the feature date: the **fair value** target |
| `value_24m`, `value_36m` | latest value on or before 24 or 36 months later, at most a year old |
| `value_24m_date`, `value_36m_date` | the date of that value (audit: always after the feature date) |
| `log_growth_24m`, `log_growth_36m` | log of later value divided by `value_now`: the **growth** targets |

A horizon that lies past the snapshot's last valuation (2026-06-12) is
missing, so the newest seasons have no growth labels.

**For the fair value model, leave out every `market_value*` and
`value_change_1y` column**: `market_value` is the target itself. The growth
model may use them all.

## Known gaps

* Süper Lig values around mid 2025 are thin: 80.9% of 2024/25 players have
  one, and growth labels for 2021 and 2022 are fewer.
* `team_ppg` and form exist only in the `match` tier.
* `reports/features_report.md` lists rows, links, values and labels per
  league season after each build.
