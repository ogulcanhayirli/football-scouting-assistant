# Decisions

The design decisions D1 to D25, moved here from the README on 29 October 2026
(prompt 21). The owner's numbered decisions 1 to 12 are in CLAUDE.md
(Decisions); D20 is decision 9, D22 decision 10, D25 decision 11. Each entry
records the choice, the reason, and what would change it.

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
IP (hashed; 40 a day too since prompt 22), 1 USD a day and 5 USD a month from the usage fields, 500
characters, reserved concurrency 2 (held back while the account's Lambda
limit is 10, which caps it meanwhile), and the Claude Console workspace's
spend limit as the hard backstop. Decision 9 is revised for the chat only:
the page stays static and the weekly run is unchanged. Since prompt 20d the
page opens on the Ask tab, and the chat reaches the legends' archetypes and
the measured greats by name or nickname, each with its label. *Would change if*
the evaluation's pass rates fell short (the chat then stays off the page),
or the visitors' use outgrew the caps.
