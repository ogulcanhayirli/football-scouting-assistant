# Feature tables

Built by `uv run scout features` from the processed season stats, match stats
and the Transfermarkt snapshot. It spends no API requests. Output under
`data/processed/features/`:

| Table | Path | Rows |
|---|---|---|
| Features | `player_season_features/league=<key>/season=<year>/data.parquet` | 16,264 |
| Labels | `player_season_labels/league=<key>/season=<year>/data.parquet` | 16,264 |
| Links | `player_links.parquet` (API Football id to Transfermarkt id) | 7,189 |
| Profiles | `player_profiles.parquet` (display only, see below) | one per link |

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
the COVID restart, and 30 June otherwise. Brazil and Argentina play calendar
year seasons, so for them the floor is 1 January of the next year (the day
after the last match when that is later: Brasileirão 2020 ended in February
2021).

Audit columns let the contract check the rule on every build:
`last_match_date` and `value_date` must not be after `feature_date`;
`prev_feature_date`, `last_comp_match_date`, `last_injury_date`,
`last_coach_change_date` and `last_transfer_date` must be before it.
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

**Cups and national teams, the 12 months before the feature date**

| Column | Meaning |
|---|---|
| `club_cup_minutes_12m`, `club_cup_matches_12m` | Champions, Europa, Conference League, Libertadores, Sudamericana |
| `club_cup_ga_per90_12m`, `club_cup_rating_12m` | goals plus assists per 90 and minute weighted rating in them |
| `ucl_minutes_12m` | Champions League minutes alone |
| `intl_minutes_12m`, `intl_matches_12m`, `intl_ga_12m` | national team: tournaments, qualifiers, Nations League, friendlies |
| `youth_minutes_12m`, `youth_matches_12m` | youth tournaments: U21 Euro, U20 and U17 World Cups, Olympics; not counted as caps |

Missing (not 0) until the data covers the whole window: European cups from
2015, South American cups from 2019 (by the league's `region`, the
confederation; no club cups are held for `north_america` and `asia`, so MLS,
Liga MX, J1, K League 1 and the Saudi Pro League always miss them), national
teams and youth tournaments from the first match of their kind in the data
(youth: the U21 Euro of March 2021). The UEFA Youth League has lineups but
no player minutes in the API and adds nothing yet. National team data has gaps the API
leaves: no Euro qualifiers at all, and no qualifiers for the 2026 World Cup,
so recent seasons undercount international minutes for everyone.

**Team strength (`team_match_stats`), league matches before the feature date**

| Column | Meaning |
|---|---|
| `team_possession`, `team_pass_accuracy` | means per match |
| `team_shots_pm`, `team_shots_against_pm`, `team_shot_share` | shots for and against per match, share of all shots |
| `team_xg_pm`, `team_xga_pm` | xG for and against per match, from 2023/24 only |

**Injuries, the 12 months before the feature date, every league and cup**

| Column | Meaning |
|---|---|
| `injury_matches_missed_12m` | matches missed injured ("Missing Fixture"); suspensions and absences left out |
| `injury_spells_12m` | separate spells: more than 30 days between missed matches starts a new one |
| `injury_source` | `injuries` or `sidelined`; audit only, not a model input |

Two sources. The `injuries` table (one row per missed fixture) is used when
the row's own league season has injury data and the window starts on or
after 6 April 2021, when the table starts. Otherwise the `sidelined` periods
fill in: the matches the row's club played (league and club cups, from the
Elo match table) while the player was out. Only periods started before the
feature date count, and an end date on or after it is never used (the
period counts as running at the feature date). Sidelined windows must lie
between July 2010 and September 2025, its coverage, and the league season
must be one the endpoint tracked: at least 30% of its rows with a spell (the
injuries table finds about 45%; untracked seasons show 0 to 10%). That keeps
the EPL from 2011, La Liga, Ligue 1 and the Eredivisie from 2012, the
Bundesliga from 2018 and a few leagues in recent seasons; elsewhere a 0
would mean "not tracked", so the columns stay missing.
How the two agree where both cover a row: `reports/injury_sources.md`.

**Coach (`coach_spells`), the row's club and league season**

| Column | Meaning |
|---|---|
| `new_coach` | 1 when a spell started after the month of the club's last league match of the previous season and by the month of its first league match of this one |
| `coach_changed_in_season` | 1 when a spell started after the month of the first league match and by the month of the last |
| `last_coach_change_date` | the latest of those starts (audit) |

Month level, because spell dates are months: an appointment in the month of
the first match counts as new for the season, not as a change during it; a
start dated after the feature date is never used. Caretakers count. From
July 2025 the API lists sitting coaches again under a new id and their full
name (J. Neestrup, then Jacob Neestrup at FC Copenhagen); a spell of the same
person (first initial and surname) while an earlier one at the club is still
running is no change. When the club's previous season is not held, the two
months before its first match stand for the off season. Missing for clubs
without a spell starting by the end of the season.

**Transfers (API Football `/transfers`), moves dated before the feature date**

| Column | Meaning |
|---|---|
| `career_moves`, `career_loans`, `career_paid_moves` | counts; a paid move has a fee in euros |
| `moves_last_2y` | moves in the two years before |
| `days_since_last_move`, `last_move_kind` | time since, and kind of, the last move |
| `last_fee_eur`, `max_fee_eur_to_date` | fee of the last move, highest fee so far |
| `on_loan` | 1 when the last move was a loan out |

Missing for players the API lists no transfers for at all.

## Label columns

| Column | Meaning |
|---|---|
| `value_now` | value at the feature date: the **fair value** target |
| `value_24m`, `value_36m` | latest value on or before 24 or 36 months later, at most a year old |
| `value_24m_date`, `value_36m_date` | the date of that value (audit: always after the feature date) |
| `log_growth_24m`, `log_growth_36m` | log of later value divided by `value_now`; the growth model that used them is retired, the bargains backtest still reads `log_growth_24m` (did the value rise?) |

A horizon that lies past the snapshot's last valuation (2026-06-12, read
from the snapshot by `TransfermarktValuationProvider.data_end`) is
missing, so the newest seasons have no growth labels.

**For the fair value model, leave out every `market_value*` and
`value_change_1y` column**: `market_value` is the target itself. Models of a
later outcome (big move, elite, step up) may use them all.

## User facing tables (decision 5)

`scout lists` writes `candidates` and `bargains` to `data/processed/lists/`
(Parquet and CSV) with `dictionary.csv`, which explains every column in one
sentence; the same dictionary closes `reports/candidates.md`, and its text
lives in `DICTIONARY` in `src/scout/lists.py`. They are model outputs, never
model inputs.

## Player profile table (display only)

`player_profiles.parquet` has one row per linked player (`af_player_id`,
`tm_player_id`) with three fields from the Transfermarkt players table:

| Column | Meaning |
|---|---|
| `contract_expiry` | contract end date; a year outside 2000 to 2040 is an entry error and missing |
| `agent` | agent or agency name; blank is missing |
| `highest_market_value_eur` | the highest value Transfermarkt ever gave him; zero, negative or above 500m is missing |
| `as_of` | the snapshot's last valuation date; the fields describe the player around then |

These describe the player **today**, not at any feature date, so they are
for display in the current season's lists only (decision 4 shows contract
expiry for bargains). They are never model inputs: the feature contract
rejects a feature table carrying any of them, and the models' column lists
exclude them (`PRESENT_ONLY_COLUMNS` in `src/scout/contracts/schemas.py`).
The point in time counterpart of the highest value is the feature
`market_value_max_to_date`.

## The present (display only, prompt 16)

`scout current` and `scout features --current` write
`data/processed/current/`, which describes today and is never a model input:

| Table | Rows (24 October 2026) | What |
|---|---|---|
| `squads/as_of=<day>.parquet` | 25,631 | every squad player of 830 clubs: team, player, age, shirt number, position (Goalkeeper, Defender, Midfielder, Attacker); contract `SQUADS` |
| `transfers/as_of=<day>.parquet` | 324,392 | every move the 830 clubs' transfer lists give, the `TRANSFERS` columns and contract |
| `current_club.parquet` | 29,042 | one club per player today (`current_club_id`, `current_club`, `current_league`, `shirt_number`, `squad_position`, `club_source`, `squad_date`); contract `CURRENT_CLUB` |
| `club_leagues.parquet` | 658 | the clubs of our leagues' current seasons and their league |
| `player_match_stats/league=<key>/season=<year>.parquet` | 4,807 matches | the season in progress, the finished seasons' columns and contract |
| `injuries/league=<key>/season=<year>.parquet` | | the season in progress, the `INJURIES` contract |
| `season_so_far.parquet` | 16,146 | the `so_far_*` block for every player with league minutes this season |

The feature contract rejects a feature table carrying any current club
column (`PRESENT_ONLY_COLUMNS`) or any `so_far_` column
(`PRESENT_ONLY_PREFIX`), and the models' column lists leave them out.

## Known gaps

* Süper Lig values around mid 2025 are thin: 80.9% of 2024/25 players have
  one, and growth labels for 2021 and 2022 are fewer.
* `team_ppg` and form exist only in the `match` tier.
* `reports/features_report.md` lists rows, links, values and labels per
  league season after each build.
