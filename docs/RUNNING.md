# Running it

The repository layout and the full command list, moved here from the README
on 29 October 2026 (prompt 21). CLAUDE.md (Commands, Layout) holds the same
with every option; docs/DEPLOY.md the weekly run, the chat and retraining.

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
$C chat-eval --dry-run    # the 72 questions from the published tables, the tools checked against them
$C chat-eval              # the questions against chat.url (needs SCOUT_CHAT_CLAUDE_KEY or ANTHROPIC_API_KEY), reports/chat_eval.md
```

Uploading to S3 (`sync up`) and deploying a stack need the owner's yes
(CLAUDE.md, rule 8); the weekly run's writes to `processed/lists/`,
`processed/reports_text/` (prompt 19) and the site bucket are the one
exception.
