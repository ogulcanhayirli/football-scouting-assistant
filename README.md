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

The page also answers **who plays like him** (decision 10): every player
has a "Similar players" panel with the ten nearest of his position group
today, and a Similar players tab searches "plays like him now" (with
filters on age, league and ability) and "looks like he did at an age"
(Rodri at 21 against today's players aged 20 to 22). Similar means the
same way of playing with the level taken out, not the same level.

An **Ask** tab (prompt 20, a prototype) answers questions in plain words
("who is the youngest centre back with an elite chance of 30%?", "who plays
like Lionel Messi at 21?"): Claude Sonnet 5.5 looks the answer up in the
published tables with four tools, and every answer shows the tool results it
was built from. It answers only from the tables on the page, says so plainly
when they cannot answer, and is capped at 20 questions an hour per visitor,
1 USD a day and 5 USD a month.

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
| Weekly run (`.github/workflows/weekly.yml`) | Built, and the stored models are in S3 (23 October); waits for the `scout-weekly` user and the GitHub secrets and variables (docs/DEPLOY.md). A clean run of the same steps took 4 minutes 14 seconds on a 4 core machine |
| Static page (`scout site`) and its bucket (`ScoutSite`) | Live since 21 October 2026 |
| Similar players (`scout embed`, `scout similar`, decision 10) | Done 24 October 2026: a learned embedding finds the same player's other season among his five nearest 19.9% of the time against 13.2% for plain style vectors (0.35% at random); in S3 since 25 October, on the page at the next publish |
| Measured greats (`scout greats`, prompts 18b and 18c) | Done 25 October 2026: StatsBomb open data (La Liga 2004/05 to 2020/21, Premier League 2003/04, 2.83 GB) mapped to our match table and checked against API Football (tackles and interceptions agree at 0.63 and 0.68 per match and are kept, the owner's call on the identity test); Barcelona's old seasons corrected for the thin peer group (the same player found 55.1% of the time on 2016/17 to 2020/21 against 28.6% uncorrected); 23 greats by age against today's players, with the calibrated elite chance, in a page tab at the next publish |
| Archetypes (`scout archetypes`, prompt 18c) | Drafted 25 October 2026, awaiting the owner's review: eight legends from the historical record (Cruyff, Gullit, Rio Ferdinand, Maldini, Zidane, Ronaldo Nazário, Baresi, Pirlo) searched among today's players; a calibration on six players we can measure lands within 15 points on half the traits (28% for a draft of 50 everywhere), so they are labelled illustrative only |
| Scout reports (`scout reports`, prompt 19) | Built 25 October 2026: Claude Haiku 5.5 writes about 130 words per player from a fact sheet of the table columns (Message Batches API, cached prefix, rewritten only when the facts change), code checks every number and name against the sheet (two retries, then the explanation written by rule; a templated player is retried on the next run once), Claude Sonnet 5.5 judges 50 a week. The first real run (25 October) cost 1.41 USD and sent 94% of the players to the template, nearly all for running over 150 words; prompt 19b fixed the length handling (90 to 170 words pass, an example report, the cut stated on a retry), and the next weekly run is its real test (about 0.65 USD expected) |
| Chat over the tables (Ask tab, `ScoutChat`, prompt 20) | Built 26 October 2026: one Lambda behind a Function URL, Claude Sonnet 5.5 with four strict tools over the published files, the caps in a small S3 object, an Ask tab on the page. Not deployed yet: the deploy waits for the owner's yes, two IAM policy updates (docs/DEPLOY.md) and a session that holds `ANTHROPIC_API_KEY`; the evaluation (50 questions) runs right after it |
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
            │ read processed/ and raw/transfermarkt/                          ▲ write processed/lists/
            ▼                                                                 │
 GitHub Actions, weekly:  features --current ─► score (stored models, no training)
                          ─► embed ─► archetypes ─► lists ─► site ─► publish ─┤
                                                                              ▼ write
                         S3 site bucket (public read, static website hosting)
                         index.html with both tables embedded, data/*.csv, data/*.parquet,
                         data/similar.json (the similar players panel, the measured
                         greats and the archetypes, loaded on demand),
                         statsbomb-logo.png (with the greats)
                                   │ read at cold start
                                   ▼
 Ask tab ── POST ──► Lambda scout-chat (Function URL, 2 at once) ──► Claude API
                     caps in S3 scout-chat-<account> (private)       (Sonnet 5.5, 4 tools)
```

Deliberately small: one scheduled job and one static page, plus one Lambda
for the Ask tab (decision 11, prompt 20, which revises decision 9 for the
chat only). The site bucket and the chat's Function URL are the only public
resources; the site bucket holds the two tables, the
dictionary, the similar players file and, with the measured greats, the
StatsBomb logo its data agreement asks for, and nothing else (`scout publish`
refuses any other file). The
infrastructure code is two CDK stacks in `infra/`: `ScoutSite`, which
creates that bucket, and `ScoutChat`, the chat's function, its role, its
private state bucket and log group; the earlier analysis stacks are archived in
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
  providers/                API Football client and adapters; Transfermarkt snapshot; StatsBomb open data
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
  embeddings.py             style vectors, the learned embedding, similar players
  greats.py, greats_check.py  the measured greats from StatsBomb open data; the check against API Football
  archetypes.py             archetypes from the record: files checked, level free vectors, calibration
  site.py, site_template.html   the static page
  chat/                     the Ask tab's chat: data, tools, prompt, guard (caps), agent, handler
  chat_eval.py              the chat's evaluation (scout chat-eval)
  publish.py, sync.py       the weekly upload; copying data to and from S3
  cli.py                    `scout <command>`
archetypes/                 legends/ and calibration/: one YAML file per player, drafted, for the owner to edit
configs/                    backfill.ultra.yaml (current), free, pro, expansion
forecasts/<date>/           registered forecasts (committed)
infra/                      ScoutSite and ScoutChat CDK stacks (own uv environment); chat/requirements.txt
                            the chat Lambda's pinned wheels; archive/ holds the retired stacks
docs/DEPLOY.md              the weekly run and the chat: setup, secrets, IAM policy, caps, cost, retraining
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
$C embed                  # every player season embedded with the stored model; today's ten nearest
$C archetypes             # archetypes/*/*.yaml reloaded and checked, vectors, calibration, reports/archetypes.md
$C lists                  # candidates and bargains, reports/candidates.md
$C reports                # written reports, checked; a sample judged (needs ANTHROPIC_API_KEY)
$C reports --dry-run      # the fact sheets and the run's tokens and cost, estimated; no request
$C site                   # data/site/index.html and the downloads
$C publish                # processed/lists/, processed/reports_text/ and the site bucket

# Retraining, by hand (about 10 GB of memory; the models are stored as they train)
$C features               # full rebuild
$C backtest               # fair value, against its baseline
$C moves                  # big move
$C scouting               # ability score, elite, step up, riser
$C embed --train          # similar players: tested against plain style vectors, refit, stored
$C forecast history       # past forecasts rebuilt and scored; their realized rates are the shown chances
$C lists && $C site

# Downloads (each spends API requests; the raw cache is never fetched twice)
$C backfill               # season totals; --level match | team | lineup | injuries | transfers
$C transfermarkt          # the snapshot, once; --refresh for a newer release

# Similar players (after embed)
$C similar "Lamine Yamal"                 # plays like him now; --age-max, --league, --ability-min
$C similar Rodri --at-age 21              # today's players aged 20 to 22 who look like he did

# The measured greats (StatsBomb open data; download once, then offline)
$C statsbomb download     # La Liga 2004/05 to 2020/21 and the Premier League 2003/04, 2.83 GB, cached
$C statsbomb build        # per match player table, checked by the contract
$C greats                 # mapping check, the peer group correction and its test, reports/greats.md

# Registered forecasts
$C forecast freeze        # copy today's lists into forecasts/<date>; never overwrites
$C forecast score forecasts/2026-10-21 --as-of 2027-06-30

# The chat (prompt 20; deploying it is in docs/DEPLOY.md)
$C chat-eval --dry-run    # the 50 questions from the published tables, the tools checked against them
$C chat-eval              # the questions against chat.url (needs ANTHROPIC_API_KEY), reports/chat_eval.md
```

Uploading to S3 (`sync up`) and deploying a stack need the owner's yes
(CLAUDE.md, rule 8); the weekly run's writes to `processed/lists/`,
`processed/reports_text/` (prompt 19) and the site bucket are the one
exception.

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
* **The measured greats before 2015/16 are sketches.** StatsBomb's open
  seasons hold Barcelona's matches (Arsenal's for 2003/04), so a great is
  measured against Barcelona and its opponents only: a thinner, biased peer
  group, with no league or team adjustment (ours starts in 2010). A
  correction learned on 2015/16 (prompt 18c) rebuilds the league from the
  opponents for Barcelona's seasons; tested on 2016/17 to 2020/21 it finds
  the same player's API Football season in the top five 55% of the time
  against 29% uncorrected, still short of the 76% a whole league season
  gets. Arsenal's 2003/04 is not corrected (MODELS.md).
* **Archetypes are illustrative only.** The eight legends without data are
  drafts from the historical record awaiting the owner's review. Drafted
  the same way for six players we can measure, the percentiles landed
  within 15 points on half the traits (the owner's bar is most), against
  28% for a draft of 50 on every trait; every result is labelled
  "archetype from the record, not measured" (MODELS.md).
* **A written report is checked for its numbers and names, not its
  meaning.** Code rejects any number that is not in the player's fact sheet
  within its rounding and any name that is not in it, but a number of the
  sheet attached to the wrong fact (another player's ability percentile)
  passes, and so does an ordinary word that opens a sentence. The weekly
  judge (Claude Sonnet 5.5, 50 reports) reads the meaning; the owner checks
  the judge on 20 of them (MODELS.md, docs/DEPLOY.md).
* **The chat is a prototype, measured on 50 questions.** Its answers come
  only from the published tables, through four tools, and every number in
  an answer must come from a tool result; code blocks an answer that repeats
  its rules. The evaluation that decides whether it works (30 questions with
  answers computed from the tables, 10 the tables cannot answer, 10 attempts
  to pull the prompt or go off topic) runs right after the first deploy;
  its pass rates go here.
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
* **StatsBomb open data** (github.com/statsbomb/open-data) supplies the
  event data behind the measured greats (La Liga 2004/05 to 2020/21,
  Barcelona's matches; the Premier League 2003/04, Arsenal's). Used under
  the StatsBomb Public Data User Agreement: research and analysis only, no
  commercial use, the data itself never passed on (it stays in the private
  cache and bucket), and every published analysis carries the StatsBomb
  logo, which the page shows next to the greats and in its footer.
  StatsBomb asks users to register at statsbomb.com/resource-centre.
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
requirement (CloudFront in front of the same bucket). Revised narrowly by
D25: one Lambda for the chat.

**D21. Score with stored models; retrain by hand.** The models that score
the newest season are saved when they are trained (LightGBM's text format
with the column list, and the ability score's fit), and the weekly run only
loads them, so a run takes about four minutes, needs 2.5 GB of memory and
cannot quietly change a model. A full feature build (9 GB) and retraining
stay manual commands. The weekly path was checked end to end: from the
tables in S3 it reproduces the registered tables of 21 October 2026 byte for
byte. *Would change if* new data arrived every week; until a live value
source exists it does not (Limits).

**D22. Similar players: style with the level taken out, a learned
embedding only if it wins (decision 10).** Each regular player season
since 2015 is a vector of per 90 numbers and on pitch shares of the team's
shots, dribbles, passes and duels, standardised within position group and
season, with both ability percentiles regressed out of its mean and spread,
so a top player's neighbours are not simply other top players (their
ability correlates 0.03 with his, against 0.57 before). A small contrastive
model (the same player in consecutive seasons as the positive pair) beat
plain cosine similarity on held out seasons, in every position group, and
was adopted; a check on real replacements (a club sells and signs the same
position within 60 days) shows clubs pick a similar style only a little
more often than chance. Search is a numpy product in memory, no vector
database. *Would change if* the tests stopped favouring the learned model
after a retrain (the stored model then holds the baseline), or richer
event data (FBref) made a better style description possible.

**D23. The measured greats: StatsBomb open data, trusted only as far as it
was checked (prompt 18b).** StatsBomb's free event data reaches back to
2004/05 for Barcelona and 2003/04 for Arsenal, which our API data cannot.
Its events are counted into our own per match table, each measure checked
against API Football on the 15,434 player matches both sources hold
(2015/16 to 2020/21): measures below a correlation of 0.7 are left out
(tackles, interceptions), and the StatsBomb seasons of 2015/16 land next to
the same player's API Football season 79% of the time. The old seasons are
measured against Barcelona and its opponents only, a thin and biased peer
group, and say so on the page; the panel shows the calibrated elite chance
next to every similarity so a style match is never mistaken for a forecast.
Prompt 18c settled two of the owner's calls: tackles and interceptions are
kept (0.63 and 0.68 per match, but with them the thin seasons find the same
player twice as often; the identity test decided), and a correction for the
opponents' bias, learned on 2015/16 per measure and position group and
tested on 2016/17 to 2020/21 (55.1% against 28.6% in the top five), is
applied to Barcelona's old seasons. *Would change if* a fuller source of
old seasons appeared, or the correction were learned for the Premier
League too (Arsenal's 2003/04 stays uncorrected).

**D24. Archetypes from the record, labelled and calibrated (prompt 18c).**
Legends with no data at all (Cruyff, Gullit, Rio Ferdinand, Maldini,
Zidane, Ronaldo Nazário, Baresi, Pirlo) are described in the style
vector's own 18 measures, each a percentile against position peers with a
reason and sources, in YAML files the owner edits after watching matches;
they go through the same level free space as a measured season. Before
trusting the method it was calibrated: six players we can measure were
drafted the same way, blind, and the drafts landed within 15 points of the
measurement on half the traits, which is not most, so the page calls the
archetypes illustrative only and labels every result "archetype from the
record, not measured". *Would change if* a second blind draft of the
calibration players (by someone who has not seen the measured table)
landed within 15 points on most traits.

**D25. A capped chat over the published tables, one Lambda (decision 11,
prompt 20).** The Ask tab sends a question to one AWS Lambda behind a
Function URL; Claude Sonnet 5.5 (effort medium) answers with four strict
tools over the very files the page publishes (search, one player's card with
his written report, similar players with the "like him at an age" form,
what a column means and cannot tell), at most six tool calls a turn, and
the answer carries the tool results it was built from. The system prompt
gives the data's dates, makes every number come from a tool result, opens
an unanswerable question with "The tables on this page do not answer that."
and refuses anything off topic with one fixed sentence; code replaces an
answer that repeats ten words of the rules. Caps: 20 messages an hour per
IP (hashed), 1 USD a day and 5 USD a month from the usage fields, 500
characters, reserved concurrency 2 (held back while the account's Lambda
limit is 10, which caps it meanwhile), and the Claude Console workspace's
spend limit as the hard backstop. Decision 9 is revised for the chat only:
the page stays static and the weekly run is unchanged. *Would change if*
the evaluation's pass rates fell short (the chat then stays off the page),
or the visitors' use outgrew the caps.
