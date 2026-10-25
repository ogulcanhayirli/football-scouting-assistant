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
   again, and refusals are never cached (D3). Answers that change from day
   to day (squads, a fresh transfer list, the fixture list and injuries of
   a season in progress) are cached per snapshot day: `as_of=<day>` in the
   cache key, never sent (prompt 16). Processed tables are rebuilt
   from the cache with zero requests. The Transfermarkt snapshot files are
   likewise downloaded once and kept.
5. **No secrets in code.** `API_FOOTBALL_KEY` comes from the environment or
   from SSM via `SCOUT_SSM_API_FOOTBALL_KEY` (`src/scout/secrets.py`); AWS
   credentials and `SCOUT_S3_BUCKET` from the environment. `.env` is
   gitignored; `.env.example` lists the names only. Never ask the owner to
   paste a key into chat.
6. **Data never goes to git.** `data/` and `reports/` are gitignored; only the
   synthetic sample under `tests/fixtures` is committed (D7). The one
   exception is `forecasts/<date>/`: registered forecasts (the two tables as
   CSV, no raw API data) are committed so their date is on record; the
   repository is private.
7. **Source errors are repaired in the adapter, never in the cache** (D8):
   an impossible value becomes missing, is logged, and gets a test. The pandera
   contracts must keep rejecting the unrepaired value.
8. **Ask the owner before** uploading to S3 and before deploying AWS stacks.
   Deployments go only through CDK under the `scout-cfn-exec` policy and the
   `scout-boundary` permissions boundary (docs/AWS_SETUP.md). The session's
   IAM user `scout-session-sync` has no delete rights. **The one exception**
   (owner, prompt 15): the automated weekly run
   (`.github/workflows/weekly.yml`, IAM user `scout-weekly`) may write
   `processed/lists/` in the data bucket and the site bucket
   (`scout-site-<account>`) without asking. Nothing else: a session still
   asks before any upload, a manual `scout publish` included.

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
$C backtest                                  # fair value model
$C moves                                     # big move model, current scores
$C elo                                       # team Elo from cached fixtures, reports/elo.md
$C scouting [--strength v2|transfer|elo|transfer_elo]  # league strength, ability, elite, step up, riser
$C ability                                  # ability versions: stability, backtest (reports/ability.md)
$C lists                                    # the two tables and reports/candidates.md; run after backtest, moves, scouting, forecast history
$C current [--as-of DAY] [--max-requests N]  # today's squads, transfers, season in progress (about 2,000 requests)
$C features --current                       # the weekly run: newest season only, earlier ones as stored, plus the season so far
$C score                                    # the weekly run: newest season with the stored models, no training
$C embed                                    # the weekly run: every row embedded with the stored model, today's ten nearest
$C embed --train                            # similar players: tests, refit, stored embedding, reports/embeddings.md (3 minutes)
$C similar "Lamine Yamal" [--age-min A --age-max B --league KEY --ability-min P]  # plays like him now
$C similar Rodri --at-age 21                # his season at 21 against today's players aged 20 to 22
$C site                                     # data/site/index.html and the downloads (the static page)
$C publish [--role scout-site-publisher]    # processed/lists/ and the site bucket ($SCOUT_SITE_BUCKET); weekly run
$C forecast freeze [--date YYYY-MM-DD]       # copy today's lists into forecasts/<date>; never overwrites
$C forecast score forecasts/2026-10-21 --as-of 2027-06-30  # promised buckets against what happened
$C forecast history                         # rebuild and score 2015 to 2023 (reports/forecast_history.md); before lists
$C forecast revise forecasts/2026-10-21 --reason "..."  # rewrite a registered folder from today's lists, date kept, reason recorded
$C forecast window-freeze [--refreeze]      # forecasts/window-2027-01 (big move and step up top 50)
$C forecast window-score                    # reports/window_window-2027-01.md, after the window
$C sync up | down [--dry-run] [--only raw|processed|reports|transfermarkt]
uv run python scripts/run_extras.py          # sidelined, coaches, events (done 17 October)
uv run python scripts/rebuild_match_seasons.py --from-s3 serie_b:2016  # re-parse partitions offline after an adapter repair
uv run python scripts/repair_names.py [--dry-run]  # the adapter's name repair on processed partitions built before it (prompt 18b)
$C statsbomb download                        # StatsBomb open data, once: La Liga 2004/05 to 2020/21, Premier League 2003/04
$C statsbomb build                           # data/processed/statsbomb/player_match_stats/, checked by the contract
$C greats                                    # mapping check against API Football, peer group vectors, the greats' table, reports/greats.md (after embed and lists)
```

Also `athena "SQL"`, `featurestore ingest | pit --as-of DATE`,
`train | tune` (SageMaker, fair value; these four need the archived stacks,
still deployed), and `--league`, `--season` filters on `backfill`. The quota guard (`src/scout/ratelimit.py`) keeps the configured
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
| Season totals | `data/processed/player_season_stats/league=<key>/season=<year>.parquet` | 327,036 rows, 37 leagues |
| Per match player stats | `data/processed/player_match_stats/...` | 3,915,740 rows (leagues, cups, youth), from 2015; 23 partitions re-parsed on 18 October after the shared id and missing attempts repairs |
| Per match team stats | `data/processed/team_match_stats/...` | 137,600 rows; xG from 2023/24 |
| Lineups | `data/processed/match_lineups/...` | 3,944,124 rows |
| Injuries | `data/processed/injuries/...` | 139,927 rows, from 2020 |
| Sidelined periods | `data/processed/sidelined/sidelined.parquet` | 88,046 injury and suspension periods for 20,791 players (all 86,250 players asked for), mostly 2010 to September 2025 (the API adds almost nothing after); a snapshot of 2026, so only periods started by the feature date are used; tracked well only in some league seasons (FEATURES.md) |
| Coach spells | `data/processed/coaches/coach_spells.parquet` | 22,383 spells, 6,549 coaches, dates mostly to the month (a few carry a day); a missing end means still in charge; from July 2025 sitting coaches are listed again under new ids |
| Match events | `data/processed/fixture_events/...` | 1,609,653 goals, cards, substitutions and VAR events for all 107,409 fixtures with player stats |
| Transfers | `data/processed/transfers/transfers.parquet` | 285,797 moves (clubs of the original 17, fetched 12 October) |
| The present (prompt 16) | `data/processed/current/` | display only, never a model input (FEATURES.md): `squads/` and `transfers/` per snapshot day (24 October: 25,631 squad players and 324,392 moves of 830 clubs), `current_club.parquet` (29,042 players), `club_leagues.parquet`, the season in progress (`player_match_stats/`, 4,807 matches of 2026/27 or 2026; `injuries/`) and `season_so_far.parquet` (16,146 players); raw under `raw/api_football/players/squads/as_of=<day>/`, `transfers/as_of=<day>/`, `fixtures/as_of=<day>/`, `injuries/as_of=<day>/` |
| Player links | `data/processed/features/player_links.parquet` | 41,224 linked players |
| Player profiles | `data/processed/features/player_profiles.parquet` | contract expiry, agent, highest value per link; display only, never a model input |
| Features | `data/processed/features/player_season_features/league=<key>/season=<year>/data.parquet` | 233,272 rows (130,631 in the original 17 leagues), 135 columns |
| Labels | `data/processed/features/player_season_labels/...` | value now, 24 and 36 months (the 24 month growth feeds the bargains backtest) |
| Team Elo | `data/processed/elo/*.parquet` | per match pre match ratings, club and league seasons |
| Model outputs | `data/processed/models/*.parquet` | backtest predictions (`backtest_<model>.parquet`), `scouting.parquet`, `big_move_current.parquet`, labels |
| Embeddings (prompt 18) | `data/processed/embeddings/` | `vectors.parquet` (93,474 regular player seasons from 2015 with match stats: keys, name, age, club, group, ability, `main`, 18 `style_*` and 48 `emb_*` columns), `neighbours.parquet` (today's 11,908 players, ten nearest each), `dictionary.csv`; written by `embed`; in S3 since 25 October (owner's yes) |
| Stored models | `data/processed/models/stored/` | the boosters behind the newest season's scores (`fair_value_fold_2024`, `fair_value_fold_2025`, `big_move_24m`, `elite`, `step_up`, `riser`; LightGBM text plus a `.json` of columns and training rows) and `ability_v2.json` (the ability fit); written by `backtest`, `moves`, `scouting`, read by `score`. Trained 21 October 2026; in S3 since 23 October (owner's yes; only these 13 files went up, and a dry run of `sync up --only processed` then found nothing to copy). Plus `embedding.json` (prompt 18, 24 October: shrinkage priors and the encoder, written by `embed --train`, read by `embed`), in S3 since 25 October (owner's yes; with `processed/embeddings/` only these 4 files went up, and a dry run then found nothing left) |
| StatsBomb raw cache (prompt 18b) | `data/raw/statsbomb/` | 1,829 files, 2.83 GB (2,825,097,646 bytes): `competitions.json`, `matches/`, `lineups/`, `events/` (904 matches, each with a `.meta.json`), `files/LICENSE.pdf` and `files/statsbomb-logo.png`; fetched 25 October 2026 from the repository's master branch; not yet in S3 |
| StatsBomb tables (prompt 18b) | `data/processed/statsbomb/` | `player_match_stats/league=<key>/season=<year>.parquet` (30,714 rows, StatsBomb ids, never mixed with API Football rows) and `vectors.parquet` (every StatsBomb season with minutes, peer group standardised and embedded); not yet in S3 |
| The measured greats (prompt 18b) | `data/processed/greats/` | `greats.parquet` (239 seasons of 23 greats, one per age, 48 `emb_*`), `greats.json` (label, attribution, excluded measures), `statsbomb-logo.png`; read by `scout site`; not yet in S3 |
| The page | `data/site/` | `index.html` (both tables embedded, 1.8 MB, 0.3 MB gzip on 24 October) and `data/` (the two tables as CSV and Parquet, `dictionary.csv`, `similar.json`); published to `scout-site-<account>` |
| User facing tables | `data/processed/lists/` | `candidates`, `bargains` (Parquet, CSV), `dictionary.csv` (see Outputs) |
| Quota state, log | `data/state/` | |

Coverage: 37 leagues in `configs/backfill.ultra.yaml`, seasons 2010 to 2025
where the API has them. The original 17: the big five, Championship, Süper
Lig, Eredivisie, Portugal, Belgium, Scotland, Denmark, Greece, Russia,
Ukraine, Brazil, Argentina. The expansion 20 (decision 6, downloaded 16 and
17 October 2026, in the features since 18 October): Austria, Switzerland,
Croatia, Serbia, Czech Republic, Norway, Sweden, Poland, Liga MX, MLS,
Colombia, Uruguay, J1, K League 1, Saudi Pro League, 2. Bundesliga, Segunda
División, Serie B, Ligue 2, Eerste Divisie. `region` is the confederation
(owner, 18 October): `europe`, `south_america`, `north_america` (MLS, Liga
MX), `asia` (J1, K League 1, Saudi Pro League); no club cups are held for the
last two, so their cup columns stay missing. 24 competitions add context
columns to league rows: UEFA and CONMEBOL club cups, national team
tournaments and qualifiers, and 5 youth competitions (`kind: youth`, own
columns; the UEFA Youth League has no player minutes and adds nothing). The
league list in the archived `infra/archive/scout_infra/data_stack.py`
matched the config when it was archived; that test is no longer run. Per league and season coverage is in
`reports/*coverage_report.md`.

Expansion gaps: no team match stats (team strength columns missing), no
injuries table (only sidelined, where tracked), transfers only for players
who also passed through a club of the original 17 leagues; Uruguay has
season totals for 2020 only; J1 2026 (a short spring season before the move
to autumn to spring) is left out by `skip_seasons`. `configs/backfill.expansion.yaml`
stays for `scripts/run_expansion.py` (order, daily cap, S3 upload, `--restore`).
Current seasons wait for the 6 November pull.

## Models

All live in `src/scout/models/` and `src/scout/scouting.py`, use LightGBM
through the native API, and model only regular players (450+ minutes).

| Model | Command | Target | Baseline | Metrics |
|---|---|---|---|---|
| Team Elo | `elo` | result of each club match (90 minutes), from the raw `/fixtures` cache | home result rate | Brier score after the fit window |
| Fair value | `backtest` | log market value at the feature date; value columns excluded | `group_median` (league, position, age band) | MAE and RMSE in log units, median percentage error, Spearman |
| Big move | `moves` | joins a top 20 club by squad value within 24 months; returns to a former club excluded | `group_rate` | AUC, average precision, share of movers in each season's top 50 |
| Elite | `scouting` | enters the 100 most valuable players within 2 seasons, from #101 to #1,000 now; trained on all of them, evaluated and listed on the elite pool (also 23 or under) | `rank_rate` (value rank band, age band) | AUC, average precision, top 50 precision |
| Step up | `scouting` | value rank at least 3 times better, ending in the top 500, within 2 seasons | `rank_rate` | as above |
| Fall back | `scouting`, `lists` | elite pool player whose value rank is at least twice as bad after 2 seasons | the pool's base rate | rate with and without each flag, rate by flag count 0, 1, 2 or more (`reports/candidates.md`) |
| Bargains | `lists` | ratio of value to the fair value prediction, ranked within position group; top 50 per season | other eligible players of the same season, ability band and age band | share whose value rose in 24 months, share who joined a club with higher Elo within 2 seasons |
| Similar players (decision 10) | `embed --train` | the same player's other season among the five nearest (identity retrieval, 2023 to 2025) | cosine on the style vectors | hit rate overall and by position group against random; replacement check (a sale and a same position signing within 60 days: signing among the departed player's ten nearest) |
| Riser (decision 3) | `scouting` | one season ability: both next seasons 20 above the best of this and last season and at the 70th percentile or better; half of team minutes now | `rank_rate` | as above, plus top 50 precision per season against the base rate |

`scouting` also estimates league strength from players who changed leagues
and builds the ability score. Since 19 October the default is version 2
(`--strength v2`, decision 8): a per 90 composite per position group (stats
and weights from the step up traits), less the transfer league effect and
the Elo team term, last two seasons pooled by minutes, shrunk by empirical
Bayes, percentile within position group and season. Version 1 (match rating,
`--strength transfer`), `elo` and `transfer_elo` stay selectable. Results as of
October 2026, from MODELS.md and `reports/`:

* Fair value is good: MAE 0.395 over all 37 leagues against 0.780 for the
  baseline (0.396 against 0.837 on the original 17, 0.393 against 0.675 on
  the new 20; 0.402 before the expansion). The growth model barely beat its
  baseline (age carried most of it) and is retired (decision 5).
* Big move: AUC 0.928 against 0.803 over 37 leagues (0.907 against 0.780 on
  the original 17, 0.905 before); 38.5% of each season's top 50 moved.
* Scouting, 37 leagues, version 2, 2 seasons ahead: elite AUC 0.881
  (baseline 0.874) on all #101 to #1,000 candidates, but 0.774 (0.771) on
  the elite pool of 23 and under, where the baseline's top 50 is slightly
  better (0.349 against 0.342); step up 0.927 (0.912). The value rank and
  age baseline carries most of it. The models rank who makes it but do not
  predict collapses.
* Riser (decision 3, with the 70th percentile floor of prompt 12): AUC 0.799 (baseline 0.610), top 50
  precision 0.093 against a base rate of 0.017 (baseline 0.027).
* Fall back (decisions 1, 2): 17.2% of the elite pool; low minutes raises
  the risk 1.40 times, value falling 1.35, injuries 1.33; ability 1.26, not
  counted (MODELS.md, `reports/candidates.md`). By flag count (prompt 14,
  2 or more merged): 15% (0 flags), 20% (1), 25% (2 or more).
* Forecast history (prompt 14, `scout forecast history`, 2015 to 2023
  rebuilt and scored two seasons on): step up, big move and fall back
  chances are calibrated within a few points; the elite chance ran high at
  the top (shown 40%, happened 30.5%). Bargains: prime top 50 value rose
  67.0% against 47.6% for the whole list, experienced 14.4% against 7.1%.
  Since prompt 14b every shown chance is the realized rate of these
  forecasts, since prompt 14c of the last five scored seasons only
  (Forecasts below); the top elite decile now shows 30%.
* Ability score stability (year to year): version 2 0.858 against 0.503 for
  version 1 and 0.537 for the rating with the Elo team term (37 leagues);
  0.656 against 0.431 two seasons apart, where pooling cannot help.
  `reports/ability.md` and MODELS.md.
* Automatic model tuning in SageMaker was run once, for the retired growth 24m model only.
* Team Elo (`src/scout/elo.py`): 163,835 club matches from 2010, home advantage
  70, K 12, league link 1.0, fitted on matches before July 2015 only; Brier
  0.161 against 0.180 for the home rate baseline. Its league strength
  correlates -0.85 with the transfer based one (Spearman -0.72). The ability
  score with the Elo terms (`scouting --strength elo`) is slightly more stable
  year to year (0.498 against 0.489); transfer league effect plus Elo team
  term is the most stable (0.537). Details in `reports/elo.md`.

* Similar players (decision 10, prompt 18): the learned embedding finds the
  same player's other season among his five nearest 19.9% of the time
  against 13.2% for cosine on the style vectors and 0.35% at random (2023
  to 2025, trained to 2022), better in every position group (goalkeepers
  barely: 5.7% against 5.0%); adopted. Level removed: a query's and his ten
  neighbours' ability percentiles correlate 0.031 (0.037 learned) against
  0.571 with the level kept. Replacement check: the signing is among the
  departed player's ten nearest 1.52% against 0.65% at random (1.02%
  baseline). `reports/embeddings.md`, MODELS.md.
* The measured greats (prompt 18b, `reports/greats.md`): StatsBomb's events
  mapped to our match table agree with API Football per player and match
  (La Liga 2015/16 to 2020/21, 15,434 player matches) at 0.726 to 0.999 for
  every kept measure; tackles (0.625) and interceptions (0.675) fall under
  the 0.7 rule and are left out of the StatsBomb vectors. A StatsBomb
  season of 2015/16 (whole league as peers) finds the same player's API
  Football season in its top five 79.1% of the time; with Barcelona and its
  opponents as peers (every season before 2015/16) 14.3% (28.6% with every
  measure kept). On 2015/16 the thin peer group alone takes a Barcelona
  regular's top five hit from 100% to 15.8% (cosine 0.68 between the two
  vectors): the pre 2015 greats are rough sketches. MODELS.md.
* Bargains (19 October, 37 leagues, seasons 2015 to 2023, experienced 30 to
  33): the top 50 gained value far more often than matched peers (prime
  63.2% against 43.3%, experienced 11.0% against 4.5%), but moved to a club
  with higher Elo no more often (39.8% against 41.8%; 18.7% against 19.1%).
  MODELS.md.

## Outputs

Decision 5: two user facing tables, written by `scout lists`
(`src/scout/lists.py`) to `data/processed/lists/` as Parquet and CSV:

* `candidates`: one row per regular player ranked #101 to #2,000 by value,
  newest season, plus the riser model's top 50 valued below #2,000 or
  without a value (prompt 14: 38 rows added, 1,830 in all on 21 October).
  Player, age, position, club, league, value rank; elite chance (decile rate
  and players, blank outside the elite pool); step up chance; fall back risk
  (flag count 0, 1 or 2 or more, rate and players, blank outside the elite
  pool); flag count and flags with raw numbers; explanation; rising (the
  riser model's top 50); big move chance (blank at a top 20 club); value,
  fair value, ability percentile. Sorted by elite decile, then step up
  decile (prompt 14c; the chance order while the rates rise with the
  decile); the added risers come last with every chance blank.
* `bargains`: the prime (1,249) and experienced (353) lists, ranked by value
  against fair value, with the shared flags, the explanation and the on
  loan and stale value tags.
* `explanation` (both tables, `src/scout/explain.py`): the three stats where
  the player is furthest from the regulars of his position group and age
  band that season, as sentences with his number and rank ("Takes 13% of
  his team's shots, top 1% of wingers aged under 21."); a neutral sentence
  when nothing is in the top or bottom 10%. Flags and explanation read one
  peer table, and a flag column is only told on the side its flag says.
* **The present** (prompt 16, `src/scout/present.py`), newest tables only:
  `club` is now `club_at_forecast`; after `league` come `current_club`,
  `current_league` (squads of the last `scout current`) and
  `moved_since_forecast` (club id today differs from the forecast's); at
  the end the season so far block, `so_far_date` to `so_far_flags`
  (minutes, share, one season ability percentile with the stored fit,
  goals and assists per 90, the flags as they stand on the pull day). On
  24 October: 619 of 1,830 candidates and 517 of 1,602 bargains moved;
  94.2% and 92.2% have minutes this season. The past forecasts of
  `forecast history` never carry these columns.
* `dictionary.csv`: one plain sentence per column of both tables (`DICTIONARY`
  in `src/scout/lists.py`; a test checks every column has one).

The one readable report is `reports/candidates.md`: both tables' top 50 and
every backtest number next to its baseline, then the dictionary.

The public face is one static page (`scout site`, `src/scout/site.py` and
`site_template.html`): both tables as JSON inside one HTML file, plain
JavaScript (no library), tabs for candidates, bargains (prime,
experienced), similar players (prompt 18) and measured greats (prompt 18b), sortable columns, filters for position, age, league and flags,
a search box, a card layout on phones, and a "how to read this" box (calibrated
odds from past seasons, not predictions beyond the market; the valuation
date; the forecast date; the five season window; club today and season so
far, as of the pull day). A moved player carries a Moved badge and shows
his club today; a "Season so far" column and a "Moved since forecast"
filter sit in both tabs. Every row has a "Similar players" button opening
a panel with the ten nearest of his position group today (age, club,
league, ability); the Similar players tab searches "plays like him now"
(filters on age, league, ability percentile) and "looks like he did at an
age" (today's players a year either side). Both read `data/similar.json`
(today's players and every earlier season of players once in the top 500
by value, vectors as 8 bit integers; 2.9 MB, 1.4 MB gzip encoded), fetched
only when first opened; opened as a local file the page says it cannot load
it. Live at
<http://scout-site-098499087060.s3-website.eu-central-1.amazonaws.com/>
since 21 October 2026. A bargain's rising tag is the riser top 50 of
`scouting.parquet`, the same set the candidates mark. The Measured greats
tab (prompt 18b) picks a great and an age and lists today's players of that
age (in the middle of the season) and his position group nearest to him,
each with two numbers side by side, the similarity and the calibrated elite
chance of the candidates table (blank outside the elite pool), an "elite
pool only" switch, the label "style similarity from box score stats is
coarse; the same player is found in the top five about one time in five;
treat lists as shortlists", the peer group caveat, and the StatsBomb logo
and credit (also in the footer); its data is a `greats` block inside
`similar.json`, and today's rows there carry `age_mid`. `scout publish`
sends only the page, the two tables with the dictionary, `similar.json`
and, with the greats, `statsbomb-logo.png` (an allowlist; without the logo
`scout site` leaves the greats out). Model
detail stays in `reports/scouting.md`, `reports/moves.md`,
`reports/backtest_fair_value.md`, `reports/ability.md` and `reports/elo.md`.

## Forecasts

* `forecasts/2026-10-21/` is the first registered forecast: the two tables
  of 21 October 2026 as CSV with `dictionary.csv`, `forecast.json` (season
  2025, feature dates 30 June 2026 and 1 January 2026 for the eight calendar
  year leagues, Transfermarkt valuations ending 12 June 2026, row counts)
  and a README of what each column promises. Never edit it by hand; a new
  forecast gets a new dated folder (`scout forecast freeze` refuses to
  overwrite). Only an owner decision on what a column promises rewrites it,
  through `scout forecast revise FOLDER --reason TEXT`: same date and rows,
  the revision, its reason and every bucket's old and new value in
  `forecast.json` and the README, the original left in git.
* **Revision 1 (prompt 14b, 21 October 2026).** The owner's calibration fix:
  every displayed chance (elite by decile, step up, big move, fall back by
  flag count) is the realized rate of the rebuilt past forecasts (built with
  the models and buckets of their season, scored two seasons on), no longer
  the in sample rate of the pooled backtest rows, which showed 40% at the
  top of the elite chance while 30.5% happened. 30 player floor on known
  outcomes, rounding to 5% and the count kept. Changed buckets: elite
  deciles 8, 9, 10 from 20%, 25%, 40% to 15%, 20%, 35%; step up decile 5
  from 5% to under 5% and decile 9 from 15% to 10%; big move and fall back
  unchanged. All realized rates rise with their neighbours. The original is
  commit e012805; the revised folder is tagged `forecast-2026-10-21`.
  Out of sample (2017 to 2023) the top elite bucket still ran high (40%
  shown, 32.5% happened) and fall back's top shown buckets came from thin
  early tables (35% shown, 23.4% happened). The top elite decile made it
  27.6% to 32.1% in each season 2020 to 2023 against 34.5% pooled, so 35%
  is likely still a few points high; weighting recent seasons is the
  owner's call. MODELS.md.
* **Revision 2 (prompt 14c, 21 October 2026).** The owner's call on that:
  every displayed chance uses a rolling window of the **last five scored
  seasons** (`WINDOW` in `src/scout/candidates.py`), so the 2025 table reads
  2019 to 2023. Floor, rounding and count kept; a bucket under the floor
  merges with its neighbour and `reports/candidates.md` says so (none did).
  Changed buckets: elite decile 1 under 5% to 5% (2.9%, above decile 2's
  0.7%, shown as it is) and decile 10 35% to 30% (31.7%); step up decile 10
  20% to 25%; big move decile 7 5% to 10%; fall back unchanged (2 or more
  flags 22.7%, still 25%); every n about 40% smaller. Candidates are now
  sorted by decile instead of the unrounded rate, which the decile 1 dip
  would have reordered; players, order, deciles, flags and bargains are
  identical to revision 1 (commit 38a8a4b). The owner tags the final
  commit. Out of sample the two rules differ only in 2022 and 2023 (five
  scored seasons exist only from then), on 140 elite and 148 big move rows
  and no fall back row; the top elite decile and the 2 or more flags
  bucket show the same under both there (36.5% and 29.6% averaged, 33.2%
  and 22.8% happened). Mean absolute gap: elite 2.0% to 1.7%, big move
  1.1% to 1.0%, step up and fall back unchanged. MODELS.md.
* **Window forecast** (prompt 16, `src/scout/window.py`):
  `forecasts/window-2027-01/`, frozen 24 October 2026, holds the big move
  model's top 50 (pool 12,832) and the step up model's top 50 among players
  ranked #101 to #2,000 (pool 1,413), both pools without players already at
  one of the 24 top clubs (`top_clubs.csv`: top 20 by Elo after their latest
  league match, top 20 by squad value in 2025), the pools as CSV, and
  `window.json` and a README with the rule: "joins a top 20 club by Elo or
  value rank between 1 January and 3 February 2027" (a transfer, loan or
  free move; not a loan return; not to the club at the freeze).
  `scout forecast window-score` reads the transfers table and every
  snapshot under `processed/current/transfers/` and writes
  `reports/window_window-2027-01.md`, each list against its pool's rate;
  interim until a snapshot after 3 February 2027 exists. **The list is
  refreshed and frozen again on 15 December 2026** (`scout forecast
  window-freeze --refreeze`, after a `scout current` pull; the October
  freeze stays in git and in `window.json`). Scoring needs a fresh
  transfer pull after the window: 830 transfer lists, about nine days on
  the free tier.
* `scout forecast score FOLDER --as-of DATE` (`src/scout/forecast.py`)
  reads a folder and the local tables up to that date and writes
  `reports/forecast_<folder>_at_<date>.md`: elite chance bucket against the
  share that entered the top 100, fall back risk against the share that fell
  back, step up and big move the same way, bargains' top 50 against the
  whole list for value rose, each with the season's base rate. Outcomes come
  from later seasons' value ranks (`scouting.parquet`) and the transfers;
  a player with no later row is unknown, not a miss. Before two seasons have
  passed the report says it is interim. The 2025 forecast is final at the
  season 2027 feature dates (1 January and 30 June 2028).
* `scout forecast history` rebuilds 2015 to 2023 in the same folder format
  under `data/processed/forecasts/history/` and scores each two seasons on
  (`reports/forecast_history.md`); the scorer agrees with the stored labels
  on 96.7% to 99.9% of rows. It runs three times: in sample rates first
  (`history/in_sample/`), whose scored buckets and outcomes go to
  `history/outcomes.parquet`; then the realized rates of the last five
  seasons up to S minus 2 (`history/<season>/`; 2015 and 2016 show none);
  then the prompt 14b rule, every season up to S minus 2
  (`history/all_seasons/<season>/`), for the old against new section. `scout lists`
  reads `outcomes.parquet`, so run it after `forecast history`.

## Version 2 directions

* Similar players, tested by finding the same player's other season among the nearest neighbours: **done 24 October (prompt 18, decision 10).**
* The measured greats from StatsBomb open data, against today's players with the calibrated elite chance: **done 25 October (prompt 18b).**
* January window forecast, published mid December and scored in February.
* A capped chat over the two tables.
* A matchup report before any match model.

## AWS

Decision 9, built 21 October 2026 (prompt 15, docs/DEPLOY.md): a weekly
GitHub Actions run (`.github/workflows/weekly.yml`, Mondays 05:17 UTC or by
hand) pulls `processed/` and `raw/transfermarkt/`, runs the free tier update
when `configs/update.free.yaml` exists (skipped with a notice until prompt 4
creates it; it then calls `scout update`, which prompt 4 must add), then
`features --current`, `score`, `embed` (prompt 18: 17 seconds, 1.3 GB),
`lists`, `site`, `publish`. There is no daily run; the neighbours of the
current rows are computed by this weekly one. It never trains.
A clean run took 254 seconds on 4 cores (pull 80, features 92, score 56,
lists 22) and 2.5 GB of memory, and rebuilt the registered tables byte for
byte. Its credentials are GitHub secrets (`AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY` of the IAM user `scout-weekly`, `API_FOOTBALL_KEY`);
the bucket names are repository variables (`SCOUT_S3_BUCKET`,
`SCOUT_SITE_BUCKET`); the first step fails naming any that is missing.

CDK app in `infra/` with its own uv environment: one stack, `ScoutSite`
(deployed 21 October): the bucket `scout-site-098499087060` with static
website hosting and public read of objects (not listing), retained on
delete, and the role `scout-site-publisher` (session user may assume it;
PutObject on the site bucket only; `scout-boundary` attached). Deploy:
`cd infra && CDK_DEFAULT_ACCOUNT=<account id> npx aws-cdk@2 deploy ScoutSite`.
The retired stacks `ScoutData` (Glue, Athena, `scout-analyst`),
`ScoutFeatureStore` and `ScoutTraining` are in `infra/archive/` with a README
on why; their tests are not run, and they stay deployed until the owner
destroys them (`athena`, `featurestore`, `train`, `tune` use them).
Retraining is manual (docs/DEPLOY.md): `features`, `backtest`, `moves`,
`scouting`, `embed --train`, `forecast history`, `lists`, then `sync up`
with the owner's yes.

## Layout

```
src/scout/  config, secrets, ratelimit, storage, sync, athena, featurestore,
            training, scouting, cli
  providers/api_football/   client, season, match, transfer adapters
  providers/transfermarkt/  snapshot download and adapter
  providers/statsbomb/      open data download (client) and the events to match stats mapping (adapter, MAPPING)
  providers/api_football/names.py  the garbled name repair (rule 7) every API Football adapter applies
  contracts/                pandera schemas for every table
  backfill/                 season, match, injury and transfer runners
  elo.py, elo_compare.py    team Elo from cached fixtures; comparison report
  flags.py                  the one warning flag set (both lists) and the bargains' context tags
  explain.py                peer table and one line explanations (both lists, same rows as the flags)
  forecast.py               registered forecasts: freeze, score, history
  window.py                 the January window forecast: freeze, score
  embeddings.py             style vectors, the learned embedding, identity and replacement tests, similar player search
  greats.py, greats_check.py  the measured greats: season rows, peer standardisation, level, the greats' table, report; the check against API Football
  season_so_far.py          the season so far block (features --current)
  present.py                club today and season so far in both tables
  backfill/current.py       scout current: squads, transfers, the season in progress
  candidates.py             decile chance tables, fall back pool and flag count rates
  bargains.py               bargains eligibility, ranking, backtest
  lists.py                  the two user facing tables, their dictionary, reports/candidates.md
  site.py, site_template.html  the static page (scout site)
  publish.py                the weekly run's uploads, with an allowlist (scout publish)
  features/                 build.py (core), context.py (cups, team, injuries, transfers), pipeline.py
  models/                   targets, backtest, estimators, moves, run, store (the stored models)
forecasts/  registered forecasts, one dated folder each (committed)
training/   SageMaker entry point and requirements
scripts/    feasibility.py, run_extras.py, rebuild_match_seasons.py, run_expansion.py, repair_names.py
tests/      unit, contracts, fixtures (synthetic)
infra/      CDK app (ScoutSite) and its tests; archive/ holds the retired stacks
docs/       DEPLOY.md (the weekly run, secrets, IAM policy, cost), AWS_SETUP.md
.github/    workflows/ci.yml (tests on pull requests and pushes to main), weekly.yml
```

## Decisions

Owner decisions of 2026-10-16. Each one line; details in the code and docs once built.

1. Fall back risk horizon: two years, the same pool and seasons as the elite target; three years is run once as a check only and not shipped. **Built 19 October:** fall back = value rank at the end of the 2 seasons at least twice as bad (no earlier definition existed); 37.7% of 7,842 candidates (2015 to 2023). At 3 years (check) 53.2%, every flag weaker.
2. List format: each candidate shows elite chance from the model score quintile (deciles since prompt 12) and fall back risk from the flag count, with the flags and their raw numbers; no bucket with fewer than 30 players, rounded to the nearest 5%, count shown. **Built 19 October** (`src/scout/candidates.py`, `candidate_profile`, `reports/candidates.md`): flag counts 0 to 3+ give 25%, 40%, 50%, 55%; elite quintiles 0%, <5%, <5%, 5%, 25%. No flag doubles the risk; the ability flag (55th percentile) has no effect under version 2 at any threshold, so it is unchanged and awaits the owner (MODELS.md). **Fixed 19 October (prompt 12):** the pool is the elite pool, #101 to #1,000 and 23 or under at the feature date (`elite_pool`; the elite target had no age limit); 2,511 candidates, fall back 17.2%. Flags: minutes x1.40, injured x1.33, value falling x1.35, ability x1.26, under the owner's 1.3, so ability left the count and is a plain column. Three flag counts 0 to 3+: 15%, 20%, 25%, 20%. The elite model trains on every #101 to #1,000 candidate (young only: AUC 0.762 against a 0.771 to 0.777 baseline) and is scored on the pool: AUC 0.774 against 0.771. Elite chance in deciles (lowest merged to 30 players; none needed): under 5% to 40%.
3. Performance risers: baseline is the best of the player's last two seasons, with a floor of half of team minutes in the current season; the list is kept only if the fixed version beats the base rate in the backtest. **Outcome (19 October): kept.** Defined on the one season version 2 score: the next season at least 20 percentiles above the best of this and last season, the season after still there, half of team minutes now (`riser_target`). Top 50 precision 0.263 against a base rate of 0.043 (baseline 0.113), better in every test season 2018 to 2023; the old definition 0.180 against 0.029. The riser list replaces the old performance list in `reports/scouting.md`; `performance` stayed in the backtest for comparison until the simplification removed it. Without the old 70th percentile floor the current list leans to low starting scores (bases near the 20th percentile, many goalkeepers and centre backs): a floor is the owner's call. **Floor added 19 October (prompt 12):** a riser also ends at the 70th percentile or better (both later seasons). 432 risers (1.7%); top 50 precision 0.093 against a base rate of 0.017 and 0.027 for the value rank and age baseline (before: 0.263, 0.043, 0.113); AUC 0.799 against 0.610. 2019 only matches its base rate.
4. Bargains: score is actual value divided by the current value model's prediction, ranked within position; ability at or above the 70th percentile within position and at least half of team minutes; split into prime (29 and under) and experienced (30 and over); contract expiry shown. **Changed 19 October (prompt 12):** experienced is 30 to 33, older players are not listed; the flags are the shared set of `src/scout/flags.py` (minutes, injured, value falling) plus two context tags on this list only, on loan and stale value; the old short record, lost place and age 33 flags are gone.
5. Simplify: two user facing tables, candidates and bargains; keep the value model, big move, elite and step up; retire the growth model; performance risers only if decision 3 passes. **Done 19 October (prompt 12):** `scout lists` writes `candidates` (1,792 rows, 25 columns) and `bargains` (1,602 rows, 21 columns) with a dictionary, and `reports/candidates.md` is the one readable report. Step up and big move chances are decile rates among past players ranked #101 to #2,000. Retired: the growth model, its `no_change` baseline and commands (nothing used its predictions, so no replacement was needed), the old performance target, the old lists in `reports/scouting.md`, `reports/bargains.md`, `scout bargains`, the bargains' own flags, `candidate_profile`, the three year fall back check. Risers stay (decision 3 passed, also with the floor). One default path, `--strength v2`; the old variants stay selectable.
6. Data before 7 November: yes; feeder leagues first (Austria, Switzerland, Croatia, Serbia, Czech Republic, Norway, Sweden, Poland, Mexico, MLS, Colombia, Uruguay, Japan, South Korea, Saudi Arabia, then the second tiers of the big five), then the sidelined endpoint per player, match events, youth competitions, coaches, and a final pull of the current season on 6 November.
7. New free sources: FBref advanced stats for the big five and the European cups from 2017/18; Elo computed from our own fixtures; ClubElo not used because its API went offline in September 2026. API Football stays for breadth.
8. Ability score version 2 is built from per 90 stats with minutes shrinkage, without the match rating, and replaces version 1 only if it is more stable year to year and predicts the targets at least as well. Rule set on 19 October: more stable, and elite and step up AUC within 0.005 of version 1 or better; else the most stable version that passes. **Outcome (19 October): adopted.** Stability 0.858 against 0.503, elite AUC 0.881 against 0.880, step up 0.927 against 0.925; `--strength v2` is the default (MODELS.md, `reports/ability.md`).
9. Serving stack: no Lambda, API Gateway, DynamoDB or Bedrock agent; a weekly GitHub Actions run publishes Parquet, CSV and one static HTML page to S3 static hosting. **Built 21 October (prompt 15):** see AWS above and docs/DEPLOY.md; the page is live and the stored models are in S3 (23 October); the weekly run waits for the owner's setup (IAM user, secrets and variables).
10. Embeddings and similar players (prompt 18, 24 October 2026). Style vectors: the version 2 per 90 inputs and the on pitch shares of the team's shots, dribbles, passes and duels, shrunk by minutes (empirical Bayes per measure and position group, one season), standardised within position group and season, with the level taken out (both ability percentiles regressed out of each measure's mean and spread within position group and season); cosine on them is the baseline. Learned: one hidden layer of 128 to 48 dimensions, InfoNCE with the same player in consecutive seasons as the positive pair and the other players of the same position group and season as negatives, settings chosen on 2021 to 2022, tested on 2023 to 2025 after training to 2022, refit on 2015 to 2025 for `stored/embedding.json`. **Outcome: adopted** (identity retrieval 19.9% against 13.2%, every position group). The replacement check is weak support only (1.52% against 0.65% at random). No vector database: a numpy search over all vectors takes 0.5 ms. MODELS.md, `reports/embeddings.md`.

## Open questions

These are unresolved or inconsistent in the repository as of 2026-10-21.
Settle them with the owner rather than guessing.

* `SCOUTING.md` does not exist; the scouting decisions are in the docstring of
  `src/scout/scouting.py` and the numbers in MODELS.md.
* `processed/` and `reports/` in S3 were synced up on 2026-10-19 (the 37
  league build: features, labels, model outputs, 23 repaired
  `player_match_stats` partitions, `reports/injury_sources.md`) and matched
  the local files then; any rebuild since needs another `sync up`.
  S3 still holds the retired growth outputs (`reports/backtest_growth_*.md`,
  `processed/models/backtest_growth_*.parquet`) and `reports/scouting.md`
  with the old lists; `sync` never deletes, and `scout-session-sync` has no
  delete rights, so removing them is the owner's call.
* The Transfermarkt dataset (dcaribou) has been paused since its 5 September
  2026 release, with valuations ending 12 June 2026. The weekly run (decision
  9) is built, but until a replacement value source exists it republishes
  the same 2025/26 tables: a season in progress (2026) has a feature date of
  30 June 2027, more than a year after the last valuation, so its rows would
  have no market value. An in progress season also needs its own feature
  date rule.
* Weekly run setup, the owner's part (docs/DEPLOY.md): create the IAM user
  `scout-weekly` with the policy there, the three GitHub secrets and two
  variables. The 13 stored model files (`processed/models/stored/`, 9 MB)
  were uploaded on 23 October with the owner's yes. The 5 USD budget alarm could not be checked or
  created: the session user has no Budgets rights, and `scout-cfn-exec`
  allows no budgets either; the commands are in docs/DEPLOY.md.
* The free tier update (prompt 4) plugs into the weekly run as
  `scout update` behind `configs/update.free.yaml`; it must read and write the
  raw cache in S3 (rule 4: the runner starts empty) and needs IAM rights on
  `raw/api_football/` and the partitions it writes, which the rule 8
  exception does not cover.
* Shared player ids: the adapter now drops the rows of the minority club when
  one id switches clubs back and forth within a league season (two players
  on one id). When the id's real owner is the minority, his rows go
  (Gabriel Barbosa's 11 Santos matches in Brasileirão 2016). Lineups and
  season totals are not repaired the same way yet.
* Decision 7 adds FBref, but README.md (Data sources and attribution) says no
  site is scraped; FBref's terms and access method need checking first.
* README.md now describes decision 9 (prompt 15); `docs/AWS_SETUP.md` still
  explains the CDK bootstrap for all `scout-*` stacks, which is still how
  `ScoutSite` deploys.
* Decision 4 shows contract expiry, which rule 1 keeps out of features because
  the snapshot describes 2026. It now lives in `player_profiles.parquet` with
  agent and highest value, for the current season's lists only; the feature
  contract and `NOT_FEATURES` both block those columns from models.
* `ScoutData`, `ScoutFeatureStore` and `ScoutTraining` are archived
  (`infra/archive/`) but still deployed; destroying them is the owner's call.
  Their Glue schema (18 October) was never redeployed, and the Feature Store
  group predates the context features.
* API gaps: no Euro qualifier data at all and no 2026 World Cup qualifiers,
  so recent international minutes are undercounted for everyone; Liga
  Profesional 2023 has no season totals (the API returns the Copa de la Liga).
* Market value targets reflect the market's opinion; coach and system effects
  are not predictable from this data.
* Ability score version 2 picks its measures from the step up traits, which
  favour stats a rising player shows (centre backs: goals, passes, shots),
  not defending as such. The goalkeeper composite is weak (passes, fouls
  drawn, duels; prior 2,100 minutes): a full goalkeeper season counts about
  60% against the position mean. A goalkeeper specific base (goals conceded against the
  team's expected) would need xG, which the API has only from 2023/24.
* The prompt 14c rebuild is in S3 (synced up 21 October with the owner's
  approval: `processed/lists/`, `processed/forecasts/history/` with
  `outcomes.parquet` and `all_seasons/`, `reports/candidates.md`,
  `reports/forecast_history.md`); a dry run after the upload found nothing
  left to copy. A fresh session needs `sync down` plus
  `raw/transfermarkt/players.csv.gz.meta.json` (the snapshot date for
  contract expiry) to rebuild the lists byte for byte; from those files the
  prompt 14b code rebuilt revision 1 byte for byte.
  `reports/scouting_elo.md` is still the 17 league Elo run.
* Elite on the young: among the 23 and under pool the elite model barely
  beats the value rank and age baseline (AUC 0.774 against 0.771) and its
  top 50 is slightly worse (0.342 against 0.349). Recorded as a limit in
  MODELS.md and README.md (prompt 14): the product's value is calibrated
  odds, risk flags and coverage of 37 leagues, not prediction beyond the
  market. The top elite decile ran high (shown 40%, happened 30.5%); the
  owner settled it in prompt 14b with realized rates and in prompt 14c with
  a five season window (now 30%, against 27.6% to 32.1% made it in each
  season 2020 to 2023); the out of sample test can barely tell the window
  from pooling every season (Forecasts).
* The ability flag would pass the 1.3 rule below the 50th percentile (x1.33)
  but not at 55 (x1.26); the threshold was not moved after looking.
  Bargains need half of team minutes, so the minutes flag never fires on
  them.
* The riser list matches its base rate in 2019 (0.02 against 0.019); every
  other test season it beats it at least three times over.
* The default branch is `main` (since prompt 15); CI runs on pull requests
  and pushes to `main`. `claude/loving-ptolemy-ghpx5n` is no longer on the
  remote.
* The club today and the season so far (prompt 16) are as of the last
  `scout current` pull (24 October 2026; synced up the same day with the
  owner's approval: 4,062 raw files, 61 processed files under `current/`
  and `lists/`, `reports/candidates.md` and the window report; a dry run
  after found nothing left). The public page was not republished from the
  session (the site bucket was outside that approval); the Monday weekly
  run publishes it. The weekly run does not pull (it
  would need about 2,000 requests and the raw cache); the 6 November pull
  should run `scout current` again, then `features --current`, `lists`,
  `site` and `sync up`. After 7 November a full pull takes about three
  weeks on the free tier.
* The `scouting.parquet` in S3 breaks ties between equal market values in a
  different order than the build that made the registered lists: rebuilt on
  24 October, 51 candidates' value ranks swap by one with a neighbour of the
  same value (Matanović and Beraldo, both €22m, #513 and #514). The order of
  rows, deciles and every chance are unchanged; a stable tie break is not
  yet in `value_ranks`. The rebuild of 25 October (prompt 18b) showed 49
  such swaps, and in the bargains 103 values differing in the 16th digit
  (400000.0000000002 against 400000.0000000001).
* The new transfer lists (830 clubs, all 37 leagues and the cups) are kept
  as snapshots under `processed/current/transfers/` and are not merged into
  `transfers/transfers.parquet`, which the features and the big move labels
  read; merging them would change the stored features of the expansion
  leagues and is a retraining decision.
* Similar players (prompt 18, built 24 October 2026) went to S3 on 25
  October with the owner's yes (`sync up --only processed`: 4 files,
  `processed/embeddings/` with its 41 MB of vectors and
  `processed/models/stored/embedding.json`; 3,321 already there; a dry run
  after found nothing left), so the weekly run can embed. The live page has
  no similar players until the next publish (the weekly run, once its setup
  is done; the site bucket was outside this approval). `scout site` writes
  `data/similar.json` saying "not available" when there are no vectors, so
  the page itself never breaks.
* Goalkeepers have weak style vectors (identity retrieval 5.7% learned
  against 5.0% for the baseline; outfield groups 11% to 30%): the same thin
  measures as the goalkeeper ability composite. A goalkeeper specific
  description needs data the API lacks (claims, sweeping, distribution
  length).
* Garbled names (prompt 18b, rule 7): `clean_name` in every API Football
  adapter decodes UTF-8 read as Latin-1 or Windows-1252 and unescapes HTML
  entities, and the contracts reject a name it would still change. Not ten
  but 741 distinct names in 379 of 1,986 processed partitions (most the
  `&apos;` of N'Diaye, O'Brien); `scripts/repair_names.py` repaired them on
  25 October, then `features --current`, `score`, `embed`, `lists`,
  `site` and `greats` ran: 9 candidates' and 3 bargains' names changed and
  nothing else (bar the tie order and float noise above). Feature rows of
  seasons before 2025 keep the old names until the next full `scout
  features` (the similar players history rows show them).
* Prompt 18b local changes await the owner's `sync up` (none was made):
  `raw/statsbomb/` (2.83 GB), `processed/statsbomb/`, `processed/greats/`
  (the weekly run needs it for the greats tab), the 379 repaired partitions,
  the rebuilt 2025 feature partitions, `models/` outputs of `score`,
  `embeddings/`, `lists/`, `reports/greats.md` and `reports/candidates.md`.
  The registered forecast folders are unchanged.
* The measured greats leave three calls to the owner: keep the 0.7 rule
  (the two excluded measures help the thin seasons' identity check, 28.6%
  against 14.3%) or keep the measures; build a correction for the
  opponents' bias learned on 2015/16, where both peer groups exist; and
  register with StatsBomb (statsbomb.com/resource-centre), as its user
  agreement asks. StatsBomb's 2015/16 La Liga is complete, so it was
  downloaded in full (380 matches) and serves the validation and the bias
  measurement; the other seasons are Barcelona's matches only.
