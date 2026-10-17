# Football Scouting Assistant

A scouting assistant for the English Premier League and the Turkish Süper Lig,
built to run on AWS for 10 to 20 USD a month. It will estimate what a player is
worth from performance data, flag players whose market value sits below that
estimate, predict how values will move over the next two to three years, find
similar players across the two leagues, and answer scouting questions in
English and Turkish through a Bedrock agent whose numbers all come from tools.

The emphasis is on engineering quality and honest evaluation: strict point in
time features, a backtest that decides between models, and a simple baseline
that is reported even when it wins.

## Status

| Part | State |
|---|---|
| Repository skeleton, CI, data contracts | Done |
| Quota aware, resumable backfill | Done, tested against a mocked API |
| Free tier proof run (one league, one season) | Done: EPL 2023 took 51 requests and 1,074 rows passed the contract; the second run made 0 requests |
| Full historical backfill | Done: all 31 past seasons (EPL 2010 to 2025, Süper Lig 2011 to 2025) complete, 1,178 requests |
| Match level player stats | Done: EPL and Süper Lig 2015 to 2025, 275,003 player match rows, 7,898 requests on the Ultra plan |
| Private S3 copy of the data | Done: `scout sync up` stored 2,387 files; a second run copies nothing |
| Feasibility check and `FEASIBILITY.md` | Done: go. Full features from 2015, basic from 2010; 88% of players linked to Transfermarkt, precision 179 of 179 on a hand check |
| Point in time feature and label tables (`FEATURES.md`) | Done: 16,264 player seasons, 87 feature columns, contracts check the point in time rule |
| Glue and Athena over the features (`infra/`, CDK) | Deployed to eu-central-1 (stack `ScoutData`); `scout athena "SQL"` queries it |
| SageMaker Feature Store (`ScoutFeatureStore` stack) | Deployed: 16,264 records, offline store (Iceberg); its point in time answers match our tables exactly |
| Expanding window backtest, baselines, first LightGBM (`MODELS.md`) | Done: fair value clearly beats the baseline; growth does not yet |
| Automatic model tuning in SageMaker (`ScoutTraining` stack) | Done for 24 month growth: 20 jobs, about 7 cents; tuned model beats the baseline narrowly (0.558 against 0.564) |
| 17 leagues, 19 cups and national team competitions, team stats, lineups, injuries, transfers | Done: about 245,000 requests on the Ultra plan; 2.2 million player match rows, 285,797 transfers; all in S3 |
| Features on all leagues plus cup, team, injury and transfer context | Done: 130,631 player seasons, 129 columns; fair value error 0.536 to 0.391 on the original EPL and Süper Lig test rows |
| Big move model (`scout moves`) | Done: AUC 0.905; 39% of each season's top 50 join a top 20 club within two years (base rate 2.3%) |
| Athena, Feature Store and tuning on the new data | Not yet: Glue schema regenerated, redeploy pending |
| Similarity search, agent, rest of the AWS stack | Not started |

## Architecture

Current scope (this repository today):

```
 API Football v3 ──► ApiFootballClient ──► raw cache (JSON, byte for byte, + metadata)
   (QuotaGuard: per minute window,              │
    daily budget, server headers)               ▼
                                         ApiFootballStatsProvider ──┐
                                                                    ├─► pandera contracts ─► Parquet
 transfermarkt-datasets ──► snapshot CSVs ──► TransfermarktValuationProvider ┘
                                                                    │
                                                    matching (DOB block + fuzzy names + club)
```

Planned AWS shape (serverless, no always on cost):

```
 CloudFront ─► API Gateway (per user rate limits) ─► Lambda: models, similarity, filters
                                                  └► Lambda: Bedrock agent (tools = the above)
 S3: raw cache, Parquet, model artifacts, embeddings    DynamoDB: player profiles, usage
 EventBridge schedule ─► Lambda: free tier update of the current season
 AWS Budgets alarm, everything in AWS CDK (Python)
```

Deliberately excluded to stay inside the budget: SageMaker real time endpoints,
OpenSearch Serverless, NAT Gateways and RDS. Models are served inside Lambda and
similarity search runs on embeddings loaded from S3.

## Layout

```
src/scout/
  config.py                 YAML settings validated with pydantic
  secrets.py                API key from the environment, then SSM Parameter Store
  ratelimit.py              QuotaGuard: per minute window and persisted daily budget
  storage.py                RawStore interface and local implementation
  providers/base.py         StatsProvider and ValuationProvider interfaces
  providers/api_football/   HTTP client, season adapter, match adapter
  providers/transfermarkt/  snapshot downloader and adapter
  contracts/                pandera schemas for the canonical tables
  matching.py               player matching between sources
  feasibility.py            analysis functions used by scripts/feasibility.py
  backfill/                 season and match runners, coverage reports
  features/                 point in time feature and label tables
  sync.py                   copy data/raw and data/processed to or from S3
  athena.py                 run SQL on the feature tables as the scout-analyst role
  featurestore.py           Feature Store ingestion and the point in time comparison
  models/                   targets, expanding window backtest, baselines, LightGBM
  cli.py                    `scout backfill | coverage | transfermarkt | sync`
configs/                    backfill.free.yaml, backfill.pro.yaml
scripts/feasibility.py      feasibility check (free tier sample, or every backfilled season)
FEASIBILITY.md              results of the feasibility check and the decisions they support
FEATURES.md                 data dictionary of the feature and label tables
MODELS.md                   targets, the backtest design and its results
tests/unit, tests/contracts, tests/fixtures (synthetic sample)
infra/                      AWS CDK app (own uv environment): Glue, Athena, query role
docs/AWS_SETUP.md           one time account setup that lets the session deploy, narrowly
```

## Running it

```bash
uv sync --all-extras
cp .env.example .env              # then set API_FOOTBALL_KEY; .env is gitignored
set -a; source .env; set +a

uv run pytest                     # unit and contract tests, no network
uv run ruff check . && uv run ruff format --check .

# Proof run on the free tier: one league, one season. Run it twice; the
# second run makes zero requests.
uv run scout --config configs/backfill.free.yaml backfill --league epl --season 2023

# Feasibility check, capped at 40 new requests
uv run python scripts/feasibility.py --config configs/backfill.free.yaml --max-requests 40

# Feasibility on every backfilled season (after the backfill and
# `scout transfermarkt`); spends no requests when the cache is complete
uv run python scripts/feasibility.py --config configs/backfill.pro.yaml --no-download --from-processed

# Full historical backfill during the Pro month
uv run scout --config configs/backfill.pro.yaml backfill

# Per match player stats (one request per match), for seasons that have them
uv run scout --config configs/backfill.ultra.yaml backfill --level match
# The same per match for team stats, lineups; per league season for injuries;
# add --scope competitions for the cups and national team competitions
uv run scout --config configs/backfill.ultra.yaml backfill --level team
uv run scout --config configs/backfill.ultra.yaml backfill --level lineup
uv run scout --config configs/backfill.ultra.yaml backfill --level injuries
# Transfer history, one request per club in the downloaded leagues (about 640)
uv run scout --config configs/backfill.ultra.yaml backfill --level transfers

# Point in time feature and label tables (needs the Transfermarkt snapshot)
uv run scout --config configs/backfill.ultra.yaml features

# Big move model: backtest, report and shortlist in reports/moves.md
uv run scout --config configs/backfill.ultra.yaml moves

# Deploy the Glue catalog and Athena workgroup (see docs/AWS_SETUP.md first)
cd infra && CDK_DEFAULT_ACCOUNT=<account id> npx aws-cdk@2 deploy ScoutData && cd ..

# Load the features into SageMaker Feature Store, then check its point in time view
uv run scout featurestore ingest
uv run scout featurestore pit --as-of 2023-01-01

# Expanding window backtest of the baselines and LightGBM, all targets
uv run scout backtest

# Run the backtest in SageMaker: one training job, or automatic model tuning
uv run scout train --target growth_24m
uv run scout tune --target growth_24m --max-jobs 20 --parallel 2
uv run scout backtest --target growth_24m --tuning-job <tuning job name>

# Query the feature tables in Athena, as the read only scout-analyst role
uv run scout athena "SELECT player_name, goals_per90 FROM player_season_features
                     WHERE league = 'epl' AND season = 2024 AND regular
                     ORDER BY goals_per90 DESC LIMIT 10"

# Copy the data to the private bucket in $SCOUT_S3_BUCKET, or restore it
# into a fresh checkout. Files already on the other side are skipped.
uv run scout sync up                # or: sync down, --dry-run, --only raw, --workers 16
```

The backfill writes `reports/coverage_report.md` and `.json`, logs to
`data/state/scout.log`, and stores Parquet under
`data/processed/player_season_stats/league=<key>/season=<year>.parquet`.

## Data sources and attribution

* **API Football** (api-football.com) supplies player season statistics. Used
  under the account holder's subscription. Raw responses stay in a private cache
  and are not redistributed.
* **transfermarkt-datasets** by dcaribou
  (github.com/dcaribou/transfermarkt-datasets) supplies players, appearances,
  market value history and transfer fees, derived from Transfermarkt (transfermarkt.com). The
  project states no license, so it is used here for non commercial purposes only
  and credited here. The snapshot ends in July 2026, and its last valuation
  is dated 2026-06-12. `scout transfermarkt --refresh` checks the published
  release and downloads only files whose ETag changed, moving the old copies
  to `data/raw/transfermarkt/archive/<date>/`; as of 2026-10-16 upstream
  updates are paused and the release of 5 September 2026 is the latest.
* No site is scraped. No club logos or badges are stored or shown, and the
  `logo` and `photo` URLs in API responses are dropped by the adapter.

This is a non commercial portfolio project.

## Decisions

Each entry records the choice, the reason, and what would change it.

**D1. Download history in one Pro month, then return to the free tier.**
Historical seasons never change, so they only need downloading once. One month
of the Pro plan (19 USD, 7,500 requests a day) covers every past season for
both leagues with room to spare; the free tier (100 a day) then keeps the
current season up to date. The backfill takes its limits from config, so
switching plans is a config change. *Would change if* the feasibility estimate
showed the backfill needs more than about half of a Pro month, for example if
match level stats turn out to be necessary.

**D2. The raw cache is the resume state.** A request is skipped when its cache
file exists. The body is written last through an atomic rename, so a crash can
never leave a half written entry. This gives resumability and idempotency
without a separate database, and processed data can always be rebuilt from the
cache without spending requests.

**D3. Refusals are errors, not data.** API Football reports plan limits, quota
exhaustion and bad keys as HTTP 200 with an `errors` field. The client
classifies those and never caches them, so a refused season can't be stored as
an empty season.

**D4. Count before sending, and only let the server tighten the budget.** The
quota guard counts a request before it goes out and persists the daily count,
so a crash can only overcount. Remaining quota headers from the server can
lower our budget but never raise it.

**D5. Adapters plus contracts at the boundary.** Each source has one adapter
that emits canonical tables, and pandera contracts check them (types, ranges,
uniqueness, and facts like shots on target never exceeding shots). Adding a
source such as Sportmonks means one new adapter that passes the same contracts.

**D6. Match players on exact date of birth first, then fuzzy names.** Date of
birth removes almost all wrong candidates, token set name scoring handles
abbreviated and long legal names, and club similarity breaks ties. Turkish
letters are mapped explicitly before transliteration. When no name scores
high enough, a single candidate with the same surname and first initial is
accepted, which handles nicknames ("Zachary" and "Zak"). Every non match
carries a reason, and precision is checked by hand on a sample, not only the
match rate.

**D7. Data lives in a private S3 bucket, never in git.** `scout sync up`
copies `data/raw` and `data/processed` to `raw/` and `processed/` in the
bucket; `scout sync down` restores them, so a new machine or session needs no
API requests. A file is skipped when size and MD5 match the S3 ETag. The sync
never deletes, the IAM user has no delete permission, and bucket versioning
keeps old copies. *Would change if* the bucket used SSE-KMS, where the ETag
is not the MD5 and every file would be copied each time.

**D8. Impossible values become missing, not guessed.** The API sometimes
sends negative counts (EPL 2015 has `lineups: -1` for Matt Jarvis, player
20439). The adapter stores any negative stat as missing and logs a warning
naming the player; the raw cache keeps the original response, so the choice
can be revisited without new requests.

**D9. Season end (30 June) as the main feature date.** Season level totals
are then enough and cost far fewer requests than match level stats.
Confirmed by the feasibility check: 99% of regular players from 2015 on have
a market value at that date. When a season runs past 30 June (2019/20 ended
on 26 July 2020), the feature date is the day after its last match instead,
so no row ever uses a match from after its feature date. *Would change if* features at other dates (for
example before the January window) became a requirement; match level stats
would then cost about 10,700 requests.

**D10. Two feature tiers instead of imputation.** Rating, shots, passes,
duels and dribbles only exist from 2015, so models use full features from
2015 and basic ones (minutes, goals, appearances, cards) back to 2010, and the
backtest compares the two. Blank counts are read as 0 when the stat is
recorded that season, because the API leaves zeros blank (see
`FEASIBILITY.md`). *Would change if* a source with older detailed stats were
added.

**D11. Fetch match level stats once, while the Ultra plan lasts.** They
cost 7,898 requests (a day on Ultra, 79 days on the free tier) and allow
features at any date and recent form from 2015. A season is probed with five
matches before it is fetched, because the API's coverage flag is wrong for
EPL 2014. Source errors (missing ids, two players sharing an id) are
repaired in the adapter and logged, never in the raw cache. *Would change
if* a later need showed the season level data was enough, in which case the
match tables simply go unused.

**D12. From 2015, playing time comes from the match data.** The season
endpoint undercounts Süper Lig minutes (90% within 5% of Transfermarkt,
against 97% for summed match minutes), worst in 2019/20 after the COVID
restart. Season totals are used only where no match data exists (2010 to
2014). *Would change if* a later API correction made the season endpoint
agree again.

**D13. AWS region eu-central-1 (Frankfurt).** Chosen by the account owner.
Bedrock model availability (Amazon Nova Lite, Claude Haiku 4.5) is checked in
the current AWS documentation before the agent is deployed; the data and
feature stack does not depend on it.

**D14. Data lake first, Feature Store second.** Features are Parquet on S3,
partitioned by league and season, catalogued in Glue and queried with
Athena. SageMaker Feature Store comes after, as a feature group over the same
table, to compare its point in time queries with ours. Point in time
correctness lives in our code either way. *Would change if* the API needed
low latency lookups that DynamoDB could not serve.

**D15. Feature Store as a mirror, not the source of truth.** The Parquet
tables stay the source for training; the feature group holds the same rows,
keyed by league and player with the feature date as event time, offline
store only. Its point in time answers were checked against ours: identical
for every player as of 2023-01-01 (4,426 rows) and 2020-07-15 (3,318 rows,
inside the COVID season, where the 2019/20 rows must not show yet). *Would
change if* the API needed online lookups; the group would then get an online
store, and Feature Store would become the serving path.

**D16. Expanding window backtest, never random folds.** Each season is
predicted by a model trained on earlier seasons only, with a gap as long as
the label horizon, training rows purged when their label was observed after
the test date, and a check on the actual dates before every fold. Random k
fold would let a model see a player's later seasons and later price levels.
The simple baseline is reported next to every model. *Would change if* never:
this is the rule the project's honesty rests on.

**D17. Context features count events by date, not by season.** Cup, national
team, injury and transfer features count what happened in the 12 months (or
the career) before the feature date, whichever league or competition it came
from, and are missing, not 0, where the source does not cover the window.
*Would change if* a source gained full history, which would only widen the
covered rows.

**D19. A big move is a new move to a top 20 club by squad value.** The
top list is recomputed every season from values known then, so it follows
the market (Brighton and Aston Villa recently) without a hand made list. A
return to a former club or a loan owner is not a big move. *Would change if*
a scout cares about a different step, for example top 50 clubs, or moves
abroad from a given league; both are a parameter of the label.

**D18. Backtest folds follow the feature date.** A row's fold is the
European season that ended last before its feature date, so a calendar year
league's season ending in December (feature date January) sits with the
European season before it. Folding by season number would make every growth
label of the European rows two seasons back look late and purge them.
