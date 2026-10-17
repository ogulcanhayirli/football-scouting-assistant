# CLAUDE.md

Orientation for a new session. Details live in README.md (decisions D1 to D19),
FEATURES.md (data dictionary), MODELS.md (models and results), FEASIBILITY.md
and docs/AWS_SETUP.md. When this file and the code disagree, the code wins;
fix this file.

## Goal and owner

* Find players likely to reach elite level (the level of Liverpool, Barcelona,
  Bayern regulars) **before the market prices it in**, plus **bargains**:
  players already very good but cheap, at any age.
* The owner is a football fan, not a professional scout. Explain results in
  football terms, show names, and question decisions openly.
* AWS budget: 10 to 20 USD a month, region eu-central-1 (D13).
* API Football is on the Ultra plan (75,000 requests a day, 450 a minute)
  **until 7 November 2026**. After that only the free tier (100 a day) is
  assumed (D1). Any download still wanted must happen before then.

## Rules for every change

1. **Point in time.** A feature row describes a player as of its
   `feature_date` and uses nothing dated after it. The feature date is the
   later of 30 June after the season and the day after the season's last
   league match; calendar year leagues (Brazil, Argentina) use 1 January
   instead of 30 June. Audit columns (`last_match_date`, `value_date`,
   `prev_feature_date`, `last_comp_match_date`, `last_injury_date`,
   `last_transfer_date`) are checked against `feature_date` by the contract in
   `src/scout/contracts/schemas.py`. Labels lie after the feature date; a
   label past the data end is missing, never stale. Transfermarkt contract
   expiry, caps and current value describe 2026 and are not used.
2. **Season by season evaluation, never random folds** (D16, D18). Each test
   season is predicted by a model trained only on earlier seasons, with a gap
   of `max(1, horizon)` seasons, purging of training rows whose label was
   observed after the test date, and `check_no_leak` on the actual dates.
   Folds come from the feature date (`fold_season` in
   `src/scout/models/backtest.py`), not the season number.
3. **A simple baseline is reported next to every model**, in the same table.
4. **Raw API responses are cached byte for byte and never downloaded twice**
   (D2). `src/scout/storage.py` writes the body last through an atomic rename,
   with a `.meta.json` sidecar; a request whose body file exists is never made
   again, and refusals are never cached (D3). Processed tables are rebuilt
   from the cache with zero requests. The Transfermarkt snapshot files are
   likewise downloaded once and kept.
5. **No secrets in code.** `API_FOOTBALL_KEY` comes from the environment or
   from SSM via `SCOUT_SSM_API_FOOTBALL_KEY` (`src/scout/secrets.py`); AWS
   credentials and `SCOUT_S3_BUCKET` from the environment. `.env` is
   gitignored; `.env.example` lists the names only. Never ask the owner to
   paste a key into chat.
6. **Data never goes to git.** `data/` and `reports/` are gitignored; only the
   synthetic sample under `tests/fixtures` is committed (D7).
7. **Source errors are repaired in the adapter, never in the cache** (D8):
   an impossible value becomes missing, is logged, and gets a test. The pandera
   contracts must keep rejecting the unrepaired value.
8. **Ask the owner before** uploading to S3 and before deploying AWS stacks.
   Deployments go only through CDK under the `scout-cfn-exec` policy and the
   `scout-boundary` permissions boundary (docs/AWS_SETUP.md). The session's
   IAM user `scout-session-sync` has no delete rights.

## Commands

```bash
uv sync --all-extras
uv run pytest                                  # everything; no network
uv run pytest -m "not contract and not live"   # what CI runs as unit tests
uv run pytest -m contract                      # data contracts on the sample
uv run ruff check . && uv run ruff format --check .
(cd infra && uv sync --group dev && uv run pytest)   # CDK app, own environment
```

Every command takes a config; the current one is `configs/backfill.ultra.yaml`
(`backfill.free.yaml` and `backfill.pro.yaml` are the earlier plans).

```bash
C="uv run scout --config configs/backfill.ultra.yaml"
$C backfill                                  # season totals (/players), resumable
$C backfill --level match                    # per match player stats
$C backfill --level team | lineup | injuries  # add --scope competitions for cups
$C backfill --level transfers                # /transfers once per club
$C transfermarkt                             # the dcaribou snapshot, once
$C transfermarkt --refresh                   # newer release? archive old files, fetch changed ones
$C features                                  # feature and label tables
$C backtest                                  # value models, all targets
$C moves                                     # big move model
$C elo                                       # team Elo from cached fixtures, reports/elo.md
$C scouting [--strength transfer|elo]        # league strength, ability, 3 targets
$C sync up | down [--dry-run] [--only raw|processed|reports]
uv run python scripts/run_extras.py          # sidelined, coaches, events (done 17 October)
```

Also `athena "SQL"`, `featurestore ingest | pit --as-of DATE`,
`train | tune --target ...` (SageMaker), and `--league`, `--season` filters on
`backfill`. The quota guard (`src/scout/ratelimit.py`) keeps the configured
safety margin (500 requests) and persists the day's count in `data/state/api_football_quota.json`;
when the budget is spent the backfill stops cleanly and the same command
resumes after 00:00 UTC. In a cloud session, run long downloads as tracked
background tasks: untracked `nohup` jobs died when the container went idle.

## Data inventory

Local paths are under the repository; `scout sync` maps `data/raw` to
`raw/`, `data/processed` to `processed/` and `reports` to `reports/` in the
bucket named by `$SCOUT_S3_BUCKET` (eu-central-1).

| What | Local path | Size or count |
|---|---|---|
| API Football raw cache | `data/raw/api_football/<endpoint>/<param>=<value>/....json` | 254,398 responses, each with a `.meta.json` |
| Transfermarkt snapshot | `data/raw/transfermarkt/*.csv.gz` | release of 2026-09-05 (latest on 2026-10-16; upstream paused), values end 2026-06-12; replaced files go to `archive/<date>/` |
| Season totals | `data/processed/player_season_stats/league=<key>/season=<year>.parquet` | 183,482 rows |
| Per match player stats | `data/processed/player_match_stats/...` | 2,226,071 rows, from 2015 |
| Per match team stats | `data/processed/team_match_stats/...` | 137,600 rows; xG from 2023/24 |
| Lineups | `data/processed/match_lineups/...` | 3,944,124 rows |
| Injuries | `data/processed/injuries/...` | 139,927 rows, from 2020 |
| Sidelined periods | `data/processed/sidelined/sidelined.parquet` | 88,046 injury and suspension periods for 20,791 players, mostly 2010 to September 2025 (the API adds almost nothing after); a snapshot of 2026, so use only periods started by the feature date |
| Coach spells | `data/processed/coaches/coach_spells.parquet` | 22,383 spells, 6,549 coaches, dates to the month; a missing end means still in charge |
| Match events | `data/processed/fixture_events/...` | 1,609,653 goals, cards, substitutions and VAR events for all 107,409 fixtures with player stats |
| Transfers | `data/processed/transfers/transfers.parquet` | 285,797 moves |
| Player links | `data/processed/features/player_links.parquet` | 33,875 linked players |
| Player profiles | `data/processed/features/player_profiles.parquet` | contract expiry, agent, highest value per link; display only, never a model input |
| Features | `data/processed/features/player_season_features/league=<key>/season=<year>/data.parquet` | 130,631 rows |
| Labels | `data/processed/features/player_season_labels/...` | value now, 24 and 36 months |
| Team Elo | `data/processed/elo/*.parquet` | per match pre match ratings, club and league seasons |
| Model outputs | `data/processed/models/*.parquet` | backtest predictions, labels |
| Quota state, log | `data/state/` | |

Coverage: 17 leagues in `configs/backfill.ultra.yaml` (the big five,
Championship, Süper Lig, Eredivisie, Portugal, Belgium, Scotland, Denmark,
Greece, Russia, Ukraine, Brazil, Argentina), seasons 2010 to 2025 where the
API has them, and 19 competitions (UEFA and CONMEBOL club cups, national team
tournaments and qualifiers) used only as context for league rows. The league
list in `infra/scout_infra/data_stack.py` must match the config (a test checks
it). Per league and season coverage is in `reports/*coverage_report.md`.

Expansion (decision 6), downloaded 16 and 17 October 2026: 20 more leagues
(feeder leagues and the second tiers of the big five) and 5 youth competitions
in `configs/backfill.expansion.yaml`, with season totals, player stats per match
and lineups; 143,554, 1,689,866 and 2,599,541 rows, per season in
`configs/expansion_rows.md`. Run with `scripts/run_expansion.py` (order, daily
cap, S3 upload after each competition, `--restore` on a fresh container). Not
yet in `backfill.ultra.yaml`, the features or Glue; `region` for North America,
Asia and Saudi Arabia is unsettled. Current seasons wait for the 6 November pull.

## Models

All live in `src/scout/models/` and `src/scout/scouting.py`, use LightGBM
through the native API, and model only regular players (450+ minutes).

| Model | Command | Target | Baseline | Metrics |
|---|---|---|---|---|
| Team Elo | `elo` | result of each club match (90 minutes), from the raw `/fixtures` cache | home result rate | Brier score after the fit window |
| Fair value | `backtest` | log market value at the feature date; value columns excluded | `group_median` (league, position, age band) | MAE and RMSE in log units, median percentage error, Spearman |
| Growth 24m, 36m | `backtest` | log of value 24 or 36 months later over value now | `no_change`, `group_median` | as above |
| Big move | `moves` | joins a top 20 club by squad value within 24 months; returns to a former club excluded | `group_rate` | AUC, average precision, share of movers in each season's top 50 |
| Elite | `scouting` | enters the 100 most valuable players within 2 seasons, from #101 to #1,000 now | none yet | AUC, average precision, top 50 precision |
| Step up | `scouting` | value rank at least 3 times better, ending in the top 500, within 2 seasons | none yet | as above |
| Performance | `scouting` | both next two seasons at the 70th ability percentile or better, average 20 above now | none yet | as above |

`scouting` also estimates league strength from players who changed leagues
and builds the ability score (rating adjusted for league and team strength,
shrunk by minutes, percentile within the position group). Results as of
October 2026, from MODELS.md and `reports/`:

* Fair value is good (MAE 0.391 on the original EPL and Süper Lig test rows,
  baseline 0.835). Growth barely beats its baseline; age carries most of it.
* Big move: AUC 0.905 against 0.780; 39% of each season's top 50 moved.
* Elite AUC 0.88, step up 0.91, performance 0.81 (backtest, 2 seasons ahead).
  The models rank who makes it but do not predict collapses.
* Automatic model tuning in SageMaker was run once, for growth 24m only.
* Team Elo (`src/scout/elo.py`): 163,835 club matches from 2010, home advantage
  70, K 12, league link 1.0, fitted on matches before July 2015 only; Brier
  0.161 against 0.180 for the home rate baseline. Its league strength
  correlates -0.85 with the transfer based one (Spearman -0.72). The ability
  score with the Elo terms (`scouting --strength elo`) is slightly more stable
  year to year (0.498 against 0.489); transfer league effect plus Elo team
  term is the most stable (0.537). Details in `reports/elo.md`.

Outputs: `reports/backtest_<target>.md`, `reports/moves.md`,
`reports/scouting.md`, predictions under `data/processed/models/`.

## AWS

CDK app in `infra/` with its own uv environment; stacks `ScoutData` (Glue
tables with partition projection, Athena workgroup `scout` with a 1 GB scan
limit, read only `scout-analyst` role), `ScoutFeatureStore` (offline store,
Iceberg) and `ScoutTraining` (roles `scout-sagemaker-training` and `scout-ml-runner`). Deploy:
`cd infra && CDK_DEFAULT_ACCOUNT=<account id> npx aws-cdk@2 deploy <Stack>`.
After a feature column change, run `uv run python scripts/export_glue_schema.py`
(the Glue schema test compares it with the pipeline) and redeploy `ScoutData`.

## Layout

```
src/scout/  config, secrets, ratelimit, storage, sync, athena, featurestore,
            training, scouting, cli
  providers/api_football/   client, season, match, transfer adapters
  providers/transfermarkt/  snapshot download and adapter
  contracts/                pandera schemas for every table
  backfill/                 season, match, injury and transfer runners
  elo.py, elo_compare.py    team Elo from cached fixtures; comparison report
  features/                 build.py (core), context.py (cups, team, injuries, transfers), pipeline.py
  models/                   targets, backtest, estimators, moves, run
training/   SageMaker entry point and requirements
scripts/    feasibility.py, export_glue_schema.py
tests/      unit, contracts, fixtures (synthetic)
infra/      CDK app and its tests
```

## Decisions

Owner decisions of 2026-10-16. Each one line; details in the code and docs once built.

1. Fall back risk horizon: two years, the same pool and seasons as the elite target; three years is run once as a check only and not shipped.
2. List format: each candidate shows elite chance from the model score quintile and fall back risk from the flag count, with the flags and their raw numbers; no bucket with fewer than 30 players, rounded to the nearest 5%, count shown.
3. Performance risers: baseline is the best of the player's last two seasons, with a floor of half of team minutes in the current season; the list is kept only if the fixed version beats the base rate in the backtest.
4. Bargains: score is actual value divided by the current value model's prediction, ranked within position; ability at or above the 70th percentile within position and at least half of team minutes; split into prime (29 and under) and experienced (30 and over); contract expiry shown.
5. Simplify: two user facing tables, candidates and bargains; keep the value model, big move, elite and step up; retire the growth model; performance risers only if decision 3 passes.
6. Data before 7 November: yes; feeder leagues first (Austria, Switzerland, Croatia, Serbia, Czech Republic, Norway, Sweden, Poland, Mexico, MLS, Colombia, Uruguay, Japan, South Korea, Saudi Arabia, then the second tiers of the big five), then the sidelined endpoint per player, match events, youth competitions, coaches, and a final pull of the current season on 6 November.
7. New free sources: FBref advanced stats for the big five and the European cups from 2017/18; Elo computed from our own fixtures; ClubElo not used because its API went offline in September 2026. API Football stays for breadth.
8. Ability score version 2 is built from per 90 stats with minutes shrinkage, without the match rating, and replaces version 1 only if it is more stable year to year and predicts the targets at least as well.
9. Serving stack: no Lambda, API Gateway, DynamoDB or Bedrock agent; a weekly GitHub Actions run publishes Parquet, CSV and one static HTML page to S3 static hosting.

## Open questions

These are unresolved or inconsistent in the repository as of 2026-10-16.
Settle them with the owner rather than guessing.

* `src/scout/scouting.py` says its decisions are argued in `SCOUTING.md`,
  which does not exist. MODELS.md does not cover the scouting targets yet, and
  the README status table does not list them.
* The three scouting targets have no baseline next to the model, against
  rule 3.
* `processed/` and `reports/` in S3 were synced on 2026-10-17 (adding the Elo
  tables, `reports/elo.md` and `reports/scouting_elo.md`) and matched the
  local files then; any rebuild since needs another `sync up`.
* Decision 7 adds FBref, but README.md (Data sources and attribution) says no
  site is scraped; FBref's terms and access method need checking first.
* Decision 9 replaces the planned AWS shape in README.md (Lambda, API Gateway,
  DynamoDB, Bedrock agent); README.md still describes the old plan.
* Decision 4 shows contract expiry, which rule 1 keeps out of features because
  the snapshot describes 2026. It now lives in `player_profiles.parquet` with
  agent and highest value, for the current season's lists only; the feature
  contract and `NOT_FEATURES` both block those columns from models.
* The Glue schema is regenerated for 129 feature columns but `ScoutData` is
  not redeployed, and the Feature Store group predates the context features.
* API gaps: no Euro qualifier data at all and no 2026 World Cup qualifiers,
  so recent international minutes are undercounted for everyone; Liga
  Profesional 2023 has no season totals (the API returns the Copa de la Liga).
* Market value targets reflect the market's opinion; coach and system effects
  are not predictable from this data.
* League and team strength for the ability score: `scouting` still defaults
  to `--strength transfer`. `reports/elo.md` has the comparison; the owner
  decides between transfer, Elo, and transfer league plus Elo team (the most
  stable, not built as a flag yet). Decision 8 also asks that a replacement
  predicts the targets at least as well, which is not yet checked.
* The `pyproject.toml` description still names only the Premier League and the
  Süper Lig.
* The work branch is `claude/gifted-brown-nwtnj0`; it is not merged into
  `main`, and CI runs on pull requests and pushes to `main`.
