# Football Scouting Assistant

Finds footballers likely to reach elite level (the level of Liverpool,
Barcelona or Bayern regulars) before the market prices it in, and bargains:
players already very good but cheap, at any age. It covers **37 leagues**
(the big five, their second tiers, the rest of Europe's main leagues, Brazil,
Argentina, Colombia, Uruguay, Liga MX, MLS, Japan, South Korea and Saudi
Arabia) and runs on AWS for well under 1 USD a month.

The result is **two tables**, published once a week with one static page:
<http://scout-site-098499087060.s3-website.eu-central-1.amazonaws.com/>

* **Candidates:** every regular player ranked #101 to #2,000 by market value
  in the newest season (1,830 rows for 2025/26, including the riser model's
  top 50 valued below #2,000). Each row has the elite chance, step up chance,
  fall back risk and big move chance, each with the number of past players it
  rests on; the warning flags with the player's own numbers; a one line
  explanation; the rising tag; value, fair value and ability percentile.
* **Bargains:** regular players at the 70th ability percentile of their
  position or better whose market value is low against the fair value model,
  in two lists: prime (29 and under, 1,249 players) and experienced (30 to
  33, 353 players), with the same flags and explanation and two context tags
  (on loan, stale value).

The emphasis is on honest evaluation: strict point in time features, season
by season backtests, a simple baseline reported next to every model, and
chances shown as the rates that past forecasts actually achieved.

## Status

| Part | State |
|---|---|
| Data: API Football raw cache, 37 leagues and 24 cups or national team competitions, seasons 2010 to 2025 | Done on the Ultra plan (254,398 cached responses); all in the private S3 bucket |
| Transfermarkt snapshot (values, transfers) | Release of 5 September 2026; valuations end 12 June 2026, upstream paused |
| Point in time feature and label tables (`FEATURES.md`) | Done: 233,272 player seasons, 135 columns, contracts check the point in time rule |
| Fair value model | Done: error 0.395 against 0.780 for the baseline (log units, 37 leagues) |
| Big move model | Done: AUC 0.928 against 0.803; 38.5% of each season's top 50 join a top 20 club within two years |
| Ability score version 2 | Done: more stable year to year (0.858 against 0.503 for version 1) |
| Elite, step up, riser models | Done: elite AUC 0.774 against 0.771 on the 23 and under pool; step up 0.927 against 0.912; riser top 50 precision 0.093 against a base rate of 0.017 |
| Fall back risk and warning flags | Done: 17.2% of the elite pool fall back; three flags, each raising the risk 1.33 to 1.40 times |
| The two tables and their dictionary (`scout lists`) | Done: chances are the realized rates of rebuilt past forecasts, last five scored seasons |
| Registered forecast (`forecasts/2026-10-21/`) | Frozen 21 October 2026, revised twice that day (both recorded); final in 2028 |
| Stored models (`scout score`) | Done: trained once by the manual commands, reproduce the registered tables byte for byte |
| Weekly run (`.github/workflows/weekly.yml`) | Built; waits for the `scout-weekly` user, the GitHub secrets and the stored models in S3 (docs/DEPLOY.md). A clean run of the same steps took 4 minutes 14 seconds on a 4 core machine |
| Static page (`scout site`) and its bucket (`ScoutSite`) | Live since 21 October 2026 |
| Free tier update of the current season | Not started: prompt 4, from 6 November; the weekly run skips it until its config exists |
| Glue and Athena, Feature Store, SageMaker training stacks | Retired by decision 9, code in `infra/archive/`; still deployed until the owner removes them |

## Architecture

```
 API Football v3 ──► raw cache (byte for byte) ──► adapters + pandera contracts ──► Parquet tables
 transfermarkt-datasets ──► snapshot CSVs ─────────────────────────────────────────────┘
                                                                                        │
 manual, on a machine with about 10 GB:                                                 ▼
   scout features ─► backtest, moves, scouting (train and store the models) ─► forecast history
                                                                                        │ sync up
                                                                                        ▼
 S3 data bucket (private): raw/, processed/ (tables, models/stored/, lists/), reports/
            │ read processed/ and raw/transfermarkt/            ▲ write processed/lists/
            ▼                                                   │
 GitHub Actions, weekly:  features --current ─► score (stored models, no training)
                          ─► lists ─► site ─► publish ──────────┤
                                                                ▼ write
                         S3 site bucket (public read, static website hosting)
                         index.html with both tables embedded, data/*.csv, data/*.parquet
```

Deliberately small: one scheduled job and one static page, no servers. The
site bucket is the only public resource; it holds the two tables and the
dictionary and nothing else (`scout publish` refuses any other file). The
only infrastructure code is the `ScoutSite` CDK stack in `infra/`, which
creates that bucket; the earlier analysis stacks are archived in
`infra/archive/` with a note on why. Setup, secrets, the IAM policy and the
cost: [docs/DEPLOY.md](docs/DEPLOY.md).

Expected monthly AWS cost, at eu-central-1 prices: the data bucket's storage
(8.7 GB of current objects, mostly the raw cache: about 0.21 USD, plus any
old versions the bucket keeps), a few cents for the weekly run and the page
(requests; the transfer stays inside AWS's free 100 GB a month), and next to
nothing for the retired stacks while they stay deployed (a Glue catalog, an
idle Athena workgroup, a small offline feature store). About 0.25 to 0.50
USD a month in all, far under the 5 USD alarm.

## Layout

```
src/scout/
  config.py, secrets.py     YAML settings (pydantic); API key from the environment or SSM
  ratelimit.py, storage.py  quota guard; raw cache, written last through an atomic rename
  providers/                API Football client and adapters; Transfermarkt snapshot
  contracts/                pandera schemas for every table
  matching.py               player matching between sources
  backfill/                 season, match, injury and transfer runners, coverage reports
  features/                 point in time feature and label tables (full, or --current)
  elo.py                    team Elo from the cached fixtures
  models/                   targets, backtest, estimators, big move, stored models (store.py)
  scouting.py               league strength, ability score, elite, step up, riser
  candidates.py, flags.py   chance tables, fall back risk, warning flags
  bargains.py, explain.py   bargains lists; one line explanations
  lists.py                  the two tables, their dictionary, reports/candidates.md
  forecast.py               registered forecasts: freeze, score, history, revise
  site.py, site_template.html   the static page
  publish.py, sync.py       the weekly upload; copying data to and from S3
  cli.py                    `scout <command>`
configs/                    backfill.ultra.yaml (current), free, pro, expansion
forecasts/<date>/           registered forecasts (committed)
infra/                      ScoutSite CDK stack (own uv environment); archive/ holds the retired stacks
docs/DEPLOY.md              the weekly run: setup, secrets, IAM policy, cost, retraining
docs/AWS_SETUP.md           one time account setup that lets a session deploy, narrowly
FEATURES.md, MODELS.md, FEASIBILITY.md   data dictionary, models and results, feasibility check
tests/unit, tests/contracts, tests/fixtures (synthetic sample)
```

## Running it

```bash
uv sync --all-extras
cp .env.example .env              # then set API_FOOTBALL_KEY; .env is gitignored
set -a; source .env; set +a

uv run pytest                     # unit and contract tests, no network
uv run ruff check . && uv run ruff format --check .

C="uv run scout --config configs/backfill.ultra.yaml"

# Restore the data from the private bucket ($SCOUT_S3_BUCKET) into a fresh checkout
$C sync down --only processed && $C sync down --only transfermarkt

# The weekly run, as GitHub Actions runs it (docs/DEPLOY.md)
$C features --current     # the newest season only, earlier ones as stored
$C score                  # the newest season with the stored models, no training
$C lists                  # candidates and bargains, reports/candidates.md
$C site                   # data/site/index.html and the downloads
$C publish                # processed/lists/ and the site bucket ($SCOUT_SITE_BUCKET)

# Retraining, by hand (about 10 GB of memory; the models are stored as they train)
$C features               # full rebuild
$C backtest               # fair value, against its baseline
$C moves                  # big move
$C scouting               # ability score, elite, step up, riser
$C forecast history       # past forecasts rebuilt and scored; their realized rates are the shown chances
$C lists && $C site

# Downloads (each spends API requests; the raw cache is never fetched twice)
$C backfill               # season totals; --level match | team | lineup | injuries | transfers
$C transfermarkt          # the snapshot, once; --refresh for a newer release

# Registered forecasts
$C forecast freeze        # copy today's lists into forecasts/<date>; never overwrites
$C forecast score forecasts/2026-10-21 --as-of 2027-06-30
```

Uploading to S3 (`sync up`) and deploying a stack need the owner's yes
(CLAUDE.md, rule 8); the weekly run's writes to `processed/lists/` and the
site bucket are the one exception.

## Limits

* **Among young players the elite model barely beats the market.** On the
  elite pool (ranked #101 to #1,000 by value, 23 or under) its AUC is 0.774
  against 0.771 for value rank plus age (prompt 12 run), and its top 50 is
  slightly worse (0.342 against 0.349). A young player's price already holds
  most of what the data says about his chance of becoming elite.
* **So the product's value is not prediction beyond the market.** It is
  calibrated odds (each chance is the realized rate of past forecasts in the
  same bucket, with how many players there were), risk flags with the player's own
  numbers, and one consistent treatment of 37 leagues, far more players than
  anyone can price by hand.
* **The top elite chance runs high.** Rebuilt forecasts of 2015 to 2023 saw
  30.5% of the players shown 40% make it. Since prompt 14b every chance shown
  is the realized rate of those forecasts, and since prompt 14c only of the
  last five scored seasons (2019 to 2023 for the 2025 table): the top elite
  decile now shows 30%, against 27.6% to 32.1% made it in each season 2020
  to 2023. The test seasons can barely tell the two rules apart, so read the
  top chance with that in mind (MODELS.md).
* **Market values freeze at June 2026.** The Transfermarkt snapshot has had
  no valuations since 12 June 2026 (CLAUDE.md, open questions). Until a new
  value source arrives, every weekly run republishes the same 2025/26
  tables; a season in progress cannot be listed, because its rows would have
  no market value.
* **The weekly run does not retrain.** It scores with the models stored on
  21 October 2026; new seasons of data need the manual retraining above.
* **The page is plain HTTP.** S3 website hosting has no HTTPS, and
  CloudFront is left out on purpose; the page holds public tables only.

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
Both buckets live there.

**D14. Data lake first, Feature Store second.** Features are Parquet on S3,
partitioned by league and season, catalogued in Glue and queried with
Athena. SageMaker Feature Store comes after, as a feature group over the same
table, to compare its point in time queries with ours. Point in time
correctness lives in our code either way. Retired by D20: the stacks are in
`infra/archive/`, still deployed until the owner removes them.

**D15. Feature Store as a mirror, not the source of truth.** The Parquet
tables stay the source for training; the feature group holds the same rows,
keyed by league and player with the feature date as event time, offline
store only. Its point in time answers were checked against ours: identical
for every player as of 2023-01-01 (4,426 rows) and 2020-07-15 (3,318 rows,
inside the COVID season, where the 2019/20 rows must not show yet). Retired
with the other analysis stacks by D20; the check it made stays recorded here.

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

**D20. Serve with a weekly job and one static page (decision 9).** A
GitHub Actions run rebuilds the newest season every Monday and puts the two
tables and one HTML page, with the data embedded and a little plain
JavaScript, into a public S3 website bucket; the data bucket stays private.
Nothing runs between the weekly runs, so nothing costs anything then. Its
IAM user may read the processed tables and write only `processed/lists/`
and the site bucket. *Would change if* the tables needed answers between
runs (a chat over the tables is a version 2 direction) or HTTPS became a
requirement (CloudFront in front of the same bucket).

**D21. Score with stored models; retrain by hand.** The models that score
the newest season are saved when they are trained (LightGBM's text format
with the column list, and the ability score's fit), and the weekly run only
loads them, so a run takes about four minutes, needs 2.5 GB of memory and
cannot quietly change a model. A full feature build (9 GB) and retraining
stay manual commands. The weekly path was checked end to end: from the
tables in S3 it reproduces the registered tables of 21 October 2026 byte for
byte. *Would change if* new data arrived every week; until a live value
source exists it does not (Limits).
