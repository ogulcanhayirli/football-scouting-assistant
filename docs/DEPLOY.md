# The weekly run, the page and the chat

Decision 9: no Lambda, API Gateway, DynamoDB or Bedrock agent. Once a week a
GitHub Actions run rebuilds the current season's two tables with the stored
models and publishes them, with one static HTML page, to a public S3 website
bucket. The data bucket stays private. Decision 11 (prompt 20) revises this
narrowly: one Lambda serves the page's Ask tab, a chat over the published
tables (The chat, below); the page stays static and the weekly run is
unchanged.

```
 GitHub Actions, .github/workflows/weekly.yml (Mondays 05:17 UTC, or Run workflow)
   1. check the secrets and variables      fails naming the one that is missing
   2. scout sync down processed, transfermarkt   data bucket, read only
   3. free tier update                     skipped until configs/update.free.yaml exists
   4. scout features --current             the newest season only, earlier ones as stored
   5. scout score                          stored models (processed/models/stored/), no training
   6. scout embed                          stored embedding: vectors, today's ten nearest each
   7. scout archetypes                     archetypes/*/*.yaml reloaded and checked, their vectors
   8. scout lists                          the two tables and their explanations
   9. scout reports                        prompt 19: written reports, checked; never blocks
  10. scout site                           index.html, similar.json, reports.json
  11. scout publish                        processed/lists/ and processed/reports_text/ in the
                                           data bucket, the site bucket
                                                    │
 S3 football-scout-data-<owner> (private)            ▼
   processed/ ── read ──►  runner  ── write ──► processed/lists/, processed/reports_text/
   raw/transfermarkt/ ──►    │      └─ write ──► S3 scout-site-<account> (public read, website)
                             │                       index.html, data/*.csv, data/*.parquet,
   Claude API (Message  ◄────┘ step 9 only,          data/similar.json, data/reports.json,
   Batches) with ANTHROPIC_API_KEY                   statsbomb-logo.png (with the greats)
```

The page is <http://scout-site-098499087060.s3-website.eu-central-1.amazonaws.com/>
(plain HTTP: S3 website endpoints have no HTTPS, and CloudFront is left out
on purpose). It holds the candidates and bargains tables, the dictionary and,
since prompt 18, `data/similar.json` for the similar players panel (today's
players and the earlier seasons of players once in the top 500 by value,
with name, age, club, league, ability and the embedding as 8 bit integers;
2.9 MB, 1.4 MB gzip encoded, loaded only when the panel opens) and, since
prompt 18b, the measured greats inside the same file (read from
`processed/greats/`, which the run pulls with the rest of `processed/`;
`scout greats` writes it by hand, the run never rebuilds it) with
`statsbomb-logo.png`, the logo the StatsBomb data agreement asks every
published analysis to carry (without the logo the greats are left out),
and, since prompt 18c, the archetypes inside the same file (step 7 rebuilds
them from the files under `archetypes/` in the repository, so the owner's
edits reach the page at the next run; their calibration reads
`processed/statsbomb/vectors.parquet` as `scout greats` last wrote it),
nothing else: `scout publish` refuses any other file (`src/scout/publish.py`).

## One time setup

Everything below runs in **AWS CloudShell** in **eu-central-1**, with an
administrator login, except the GitHub part. Never paste a key into a chat or
a file in the repository: copy it from CloudShell straight into GitHub.

### 1. The site bucket (done 21 October 2026)

The `ScoutSite` stack (`infra/`, CDK, under `scout-cfn-exec` and
`scout-boundary`) holds the bucket `scout-site-098499087060` with static
website hosting, a bucket policy that lets anyone read objects (not list
them), and the role `scout-site-publisher`, which the session user may
assume to republish by hand (`scout publish --role scout-site-publisher`).
To redeploy after a change:

```bash
cd infra && CDK_DEFAULT_ACCOUNT=<account id> npx aws-cdk@2 deploy ScoutSite
```

If account level S3 Block Public Access is ever switched on, the page stops
being readable; the deploy of 21 October succeeded, so it is off today.

### 2. An IAM user for the weekly run

`scout-weekly` may read the processed tables and the Transfermarkt snapshot,
and write only `processed/lists/`, `processed/reports_text/` (prompt 19: the
written reports, kept so the next run reuses every one whose facts did not
change) and the site bucket. It cannot delete anything, and it cannot touch
`raw/api_football/`, `reports/` or any other bucket.

```bash
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
DATA=football-scout-data-ogulcanhayirli
SITE=scout-site-${ACCOUNT}

cat > scout-weekly.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListTheReadPrefixes",
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::${DATA}",
      "Condition": {"StringLike": {"s3:prefix": ["processed/*", "raw/transfermarkt/*"]}}
    },
    {
      "Sid": "ReadTablesAndSnapshot",
      "Effect": "Allow",
      "Action": "s3:GetObject",
      "Resource": [
        "arn:aws:s3:::${DATA}/processed/*",
        "arn:aws:s3:::${DATA}/raw/transfermarkt/*"
      ]
    },
    {
      "Sid": "WriteTheLists",
      "Effect": "Allow",
      "Action": "s3:PutObject",
      "Resource": "arn:aws:s3:::${DATA}/processed/lists/*"
    },
    {
      "Sid": "KeepTheReports",
      "Effect": "Allow",
      "Action": "s3:PutObject",
      "Resource": "arn:aws:s3:::${DATA}/processed/reports_text/*"
    },
    {
      "Sid": "WriteTheSite",
      "Effect": "Allow",
      "Action": "s3:PutObject",
      "Resource": "arn:aws:s3:::${SITE}/*"
    }
  ]
}
EOF

aws iam create-user --user-name scout-weekly
aws iam put-user-policy --user-name scout-weekly --policy-name scout-weekly \
  --policy-document file://scout-weekly.json
aws iam create-access-key --user-name scout-weekly
```

The last command prints `AccessKeyId` and `SecretAccessKey` once. Put them
into GitHub (next step) and clear the CloudShell screen.

If `scout-weekly` was created before prompt 19, run the `put-user-policy`
line again with the file above to add `KeepTheReports`. **Run 8 (25 October)
had no such statement**: its publish logged `AccessDenied` on
`processed/reports_text/` and kept 0 files, so the next run starts from
nothing again, and the rule that retries a templated player only once
(prompt 19b) cannot hold until the statement is there. Without it the run
still publishes; `scout publish` logs a warning that `processed/reports_text/`
was not kept, and the next run writes again every report it cannot reuse.

### 3. GitHub secrets and variables

In the repository: Settings, Secrets and variables, Actions.

| Kind | Name | Value |
|---|---|---|
| Secret | `AWS_ACCESS_KEY_ID` | `AccessKeyId` of `scout-weekly` |
| Secret | `AWS_SECRET_ACCESS_KEY` | `SecretAccessKey` of `scout-weekly` |
| Secret | `API_FOOTBALL_KEY` | the API Football key (used by the free tier update only) |
| Secret | `ANTHROPIC_API_KEY` | a Claude API key for the written reports (prompt 19); optional: without it the step logs one line and the page shows the explanations written by rule |
| Variable | `SCOUT_S3_BUCKET` | `football-scout-data-ogulcanhayirli` |
| Variable | `SCOUT_SITE_BUCKET` | `scout-site-098499087060` |

The bucket names are variables, not secrets, so the logs can show the page
address. The first step of the run fails with the name of any secret or
variable that is missing, except `ANTHROPIC_API_KEY`, which is optional.

The Claude API key comes from the Claude Console (API keys). Give it its own
workspace with a monthly spend limit (a few dollars is plenty, see The scout
reports below) so a mistake can never cost more than that.

### 4. The stored models in S3 (done 23 October 2026)

The run scores with the models in `processed/models/stored/` and never
trains. They were trained on 21 October 2026, reproduce the registered
tables byte for byte, and were uploaded on 23 October with the owner's yes:
13 files, 9 MB, nothing else. The similar players embedding
(`embedding.json`, prompt 18, trained 24 October 2026) and
`processed/embeddings/` followed on 25 October, also with the owner's yes
(4 files). If `embedding.json` is ever missing, step 6 of the run stops with
a message naming it. After a manual retrain, upload them again the same way,
with the owner's yes (rule 8):

```bash
uv run scout --config configs/backfill.ultra.yaml sync up --only processed
```

If they are ever missing, step 5 of the run fails with a message naming the
missing file.

### 5. A budget alarm at 5 USD

The session user cannot read or create budgets, so whether one exists was
not checked. To look, then to create one if there is none:

```bash
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
aws budgets describe-budgets --account-id "$ACCOUNT" \
  --query 'Budgets[].[BudgetName,BudgetLimit.Amount,BudgetLimit.Unit]' --output table

cat > budget.json <<EOF
{"BudgetName": "scout-5-usd", "BudgetLimit": {"Amount": "5", "Unit": "USD"},
 "TimeUnit": "MONTHLY", "BudgetType": "COST"}
EOF
EMAIL="<your email>"
cat > alerts.json <<EOF
[
  {"Notification": {"NotificationType": "ACTUAL", "ComparisonOperator": "GREATER_THAN",
                    "Threshold": 80, "ThresholdType": "PERCENTAGE"},
   "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "${EMAIL}"}]},
  {"Notification": {"NotificationType": "FORECASTED", "ComparisonOperator": "GREATER_THAN",
                    "Threshold": 100, "ThresholdType": "PERCENTAGE"},
   "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "${EMAIL}"}]}
]
EOF
aws budgets create-budget --account-id "$ACCOUNT" --budget file://budget.json \
  --notifications-with-subscribers file://alerts.json
```

It mails at 4 USD spent and when the month is forecast to pass 5 USD.

### 6. The first run

Actions, Weekly lists and page, Run workflow. The run's summary shows its
runtime and the page address.

## Runtime and cost

Measured on 21 October 2026 with the same commands in a clean copy of the
repository and an empty `data/` folder (4 cores, 15 GB):

| Step | Seconds |
|---|---|
| install (uv cache warm) | under 1 |
| pull `processed/` (3,253 files, 357 MB) | 76 |
| pull the Transfermarkt snapshot (65 MB) | 4 |
| `features --current` | 92 |
| `score` | 56 |
| `embed` (measured 24 October, 1.3 GB of memory) | 17 |
| `archetypes` (measured 25 October) | 21 |
| `lists` | 22 |
| `site` | 2 |
| `publish` (dry run) | 2 |
| **total** | **254** (271 with `embed`, 292 with `archetypes`) |

The tables it built were byte for byte those of the registered forecast.
A GitHub runner has 2 cores, so expect about 6 to 8 minutes there, plus a
minute to install. The heaviest step is `features --current` (2.5 GB of
memory);
a full `scout features` needs about 9 GB and 10 minutes, more than a free
GitHub runner for a private repository has (2 cores, 7 GB), which is one
reason the weekly run rebuilds the current season only.

GitHub Free gives a private repository 2,000 Actions minutes a month; four or
five weekly runs use well under 50 of them.

Expected AWS cost of the weekly run and the page, eu-central-1 prices:

| Item | Monthly amount | Cost |
|---|---|---|
| Site bucket storage | under 1 MB | 0.00 USD |
| Weekly pull of `processed/` | about 3,300 GET requests and 0.4 GB per run, about 14,000 GETs and 1.8 GB a month | 0.01 USD (transfer out is inside AWS's free 100 GB a month) |
| Page views | 0.3 MB each (stored gzip encoded); 10,000 views are 3 GB; the similar players file adds 1.4 MB when a viewer opens it | under 0.01 USD for the GETs, transfer inside the free 100 GB |
| Uploads | 13 PUT requests a run | 0.00 USD |
| Stored models | 9 MB in the data bucket, plus the 0.1 MB embedding | 0.00 USD |

So the run and the page add a few cents at most. The rest of the bill is
unchanged: the data bucket's 8.7 GB (about 0.21 USD a month, plus any old
versions it keeps) and the retired stacks while they stay deployed (next to
nothing). About 0.25 to 0.50 USD a month in all.

## The scout reports (prompt 19)

Step 9 runs `scout reports` (`src/scout/reports_text.py`): for every player of
the two tables (2,734 on the 24 October tables: 1,830 candidates and 1,602
bargains, 698 in both) a fact sheet is built from the table columns, Claude
Haiku 5.5 writes about 130 words from it (never more than 150 asked; 90 to
170 pass the check, prompt 19b) through the Message Batches API, code checks
every number and proper noun against the sheet, a failing report goes back
with the findings, its word count and the cut needed, at most twice, and
after that the explanation written by rule is shown. A passed report is
written again only when its fact sheet changed (`fact_hash`); otherwise the
stored one is reused. A player whose report ended on the template is asked
for again on the next run, once; after two runs in a row on the template he
is left to the explanation written by rule until the prompt version or the
writer model changes (`template_runs`), so failures are not paid for every
week. Each run, Claude
Sonnet 5.5 judges 50 reports not judged before (structured output: faithful
with the offending passage, hedging adequate, readability 1 to 5).

* **The step never blocks the page.** Without the `ANTHROPIC_API_KEY` secret it
  logs one line and does nothing; any failure is ignored (`continue-on-error`);
  batches still running after `max_run_minutes` (100) are cancelled and their
  players show the explanation this week.
* **Hard token cap** (`reports_text.max_tokens_per_run` in
  `configs/backfill.ultra.yaml`, 30 million): before a batch is sent, each
  request's worst case (its input counted at one token per two bytes, which
  is above any real count, plus its `max_tokens`) is added to what the run
  has used, and requests that could pass the cap are held back until the
  next run. The judge's worst case is kept free first. A full first round
  is about 25.9 million tokens in the worst case since prompt 19b (the
  writer's `max_tokens` went from 2,000 to 6,000: thinking counts toward it,
  and 7 requests of run 8 were cut off at 2,000). If nearly every report
  failed again as in run 8, the retry rounds would reach the cap and part
  of them would be held back: those players show the explanation that week
  and are written on the next run, so the cap also bounds what a failed fix
  can cost (a fake client run on the 24 October tables held back 771 of
  2,734 that way).
* **Cost line.** The run's cost is computed from the usage fields of every
  reply (input, cache writes at 2 times for the one hour cache, cache reads at
  0.1 times, output; half price through the Message Batches API) and printed
  in the run's summary as one line, `Reports: <n> written (<n> failed the
  check first, <n> still after retries, <n> held), <n> reused, <n> judged;
  cost $<total> (writer $<x>, judge $<y>); <tokens> of 30,000,000 tokens.`
  Every run also appends its numbers to `processed/reports_text/runs.jsonl`.

What it costs. The estimates below were made before any real run, from the
sizes of the 24 October fact sheets (`scout reports --dry-run`): the fixed
prefix about 1,240 tokens (1,570 since prompt 19b, with the example report),
a sheet about 490, a report about 220 plus about 230 of thinking at low
effort, a judge verdict about 800 with its thinking. Rates: Claude Haiku 5.5
0.10 and 0.50 USD per million input and output tokens, Claude Sonnet 5.5 2
and 10, both halved in a batch. The first real run (weekly.yml run 8, 25
October, commit 6ef72a8) stands next to them.

| Run | Estimate | Run 8 (real) |
|---|---|---|
| Full regeneration, writer, first tries (2,734 reports, prefix read from the cache) | about 0.40 USD | about 0.45 USD (the writer's 1.33 USD over 7,994 requests) |
| The same with no cache hit at all (every request writes the prefix) | about 0.70 USD | |
| Retries | about 0.10 USD (one report in five failing once) | about 0.90 USD: 2,692 failed the first try and 2,568 the second retry, so 5,260 retries |
| Judge, 50 reports | about 0.30 USD | 0.08 USD |
| **Full regeneration with the judge** | **about 0.70 to 1.10 USD** | **1.41 USD**, 21.2 million of the 30 million tokens |
| **Weekly run, nothing changed** (today: the tables are rebuilt byte for byte until a `scout current` pull) | **about 0.30 USD** (the judge only) | about 0.10 USD (the judge's real cost) |
| Weekly run after a pull that moves the season so far of most players | about a full regeneration | |

Why run 8 cost more: 98% of the reports failed the length rule of 120 to
150 words on the first try and 94% still after two retries, nearly all on
length alone (the reports ran 151 to 283 words on
their last try, median 177; a retry told "it must have 120 to 150" did not
shorten them), so almost every report was asked for three times. The
check's own word count is never above a plain whitespace count, so the
reports were long, not miscounted. Prompt 19b asks for about 130 with an
example of that length, tells a retry its count and the cut, passes 90 to
170, and stops retrying a templated player after two runs. Of run 8's 2,502
reports that failed on length alone, 839 (34%) would pass at 170; the rest
depend on the new prompt.

**Expected cost of the next run** (prompt 19b bumps the prompt version, so
every report is written again): first tries about 0.45 USD as in run 8, the
judge about 0.10, and retries in proportion to the failures, about 0.17 USD
per thousand retries. If one report in five fails the first try and one in
twenty the retry, about 0.65 USD; if the prompt fix does not hold and the
reports fail as in run 8, at most about 1.45 USD (the cap holds part of the retries back) (and the run after that
retries only the templated players, once). After that, about 0.10 USD a
week while nothing changes, so well under 1 USD a month, billed by
Anthropic, not AWS.

For the owner, each run writes `reports/reports_text.md` (the run in numbers,
the judge's verdicts, and 20 of the judged reports with their facts, the ones
the judge faulted first) and `reports/judge_sample.csv` with the same 20 rows.
Write yes or no in `owner_agrees` and append the rows to `judge/agreement.csv`
in the repository; the page's "how to read this" box shows the agreement rate
from the next run on. The weekly run keeps both files as the artifact
`scout-reports-judged` of the run (Actions, the run, Artifacts; 90 days); a
session running `scout reports` writes them under `reports/`. Every verdict
also stays in `processed/reports_text/verdicts.jsonl`.

## The chat (prompt 20)

The page's Ask tab posts a fan's question to one AWS Lambda behind a public
Function URL (no API Gateway). The Lambda reads the published files from the
site bucket at cold start (the page's data block, the two tables, the
dictionary, `similar.json`, `reports.json`; once an hour it asks whether the
page changed and reloads), and Claude Sonnet 5.5 answers with four tools over
them: `search_players`, `player_card`, `similar_players`, `how_to_read`
(`src/scout/chat/`). The answer comes back with the tool results it was built
from, which the page shows under it ("From the tables").

```
 page (Ask tab) ── POST {message, history} ──► Function URL (CORS: the site only)
                                                  │
                                       Lambda scout-chat (Python 3.12, arm64, 512 MB,
                                       180 s, at most 2 at once)
                     read index.html, data/* ◄────┤────► Claude API (Sonnet 5.5, 4 tools,
                     (site bucket)                │        at most 6 tool calls a turn)
                                                  ▼
                              S3 scout-chat-<account> (private): state/chat.json,
                              the caps' counters (hashed IPs this hour, USD today
                              and this month)
```

The stack is `ScoutChat` (`infra/scout_infra/chat_stack.py`): the function
`scout-chat` (no reserved concurrency for now, see below), its role `scout-chat-lambda` under
`scout-boundary` (read `index.html` and `data/*` in the site bucket, read and
write `state/*` in the chat bucket, write its own log group), the private
bucket `scout-chat-<account>` (retained on delete) and the log group
`/aws/lambda/scout-chat` (30 days; it never holds a message, an answer, the
key or the prompt, only the outcome, tool names, tokens, cost and seconds).
The code folder is built by `infra/scout_infra/chat_bundle.py` before every
synth: `src/scout/chat/` and the pinned wheels of
`infra/chat/requirements.txt` (anthropic and its dependencies, 27 MB; boto3
is the runtime's own).

### The caps

| Cap | Value | What happens |
|---|---|---|
| Per visitor | 20 messages an hour by IP (the state keeps a salted hash, for the current hour only) | the fixed reply "Sorry, you have reached the limit of 20 questions an hour..." |
| Per day | 1 USD, computed from the usage fields of every reply | the fixed reply "Sorry, the chat has used up its budget for now..." |
| Per month | 5 USD, the same way | the same reply |
| Message length | 500 characters | "Please keep your question under 500 characters." (the page stops longer ones first) |
| Tool calls | 6 a turn, then the model must answer | |
| Concurrency | the account's Lambda limit of 10 for now (reserved concurrency 2 once the quota is raised) | the Function URL answers 429; the page says the chat is busy |

The prices are those of Claude Sonnet 5.5 (2 USD per million input tokens, 10
per million output, cache writes 1.25 times and reads 0.1 times the input
rate; `PRICES` in `src/scout/chat/agent.py`). The budget is checked before
each model call, not reserved ahead, so turns running at once can pass a cap
by what those turns cost (a few cents each; with up to 10 at once, a few tens
of cents at worst). **The hard backstop is the
spend limit of the Claude Console workspace the key belongs to**: give the
chat its own key in its own workspace (Console, Workspaces, then Limits)
with a monthly limit of 10 USD (the 5 USD of visitors plus room for an
evaluation run or two), so no bug in the caps can cost more.

The evaluation (`scout chat-eval`, below) sends a token derived from the key
(an HMAC; nothing extra is stored anywhere): its questions skip the per
visitor limit and count against their own cap of 5 USD a day, so a run never
uses the visitors' budget.

### One time setup (the owner, CloudShell)

The deployment policy `scout-cfn-exec` and the boundary `scout-boundary`
(docs/AWS_SETUP.md) allow no Lambda functions, Function URLs or Lambda log
groups. This adds exactly those, for `scout-*` names only, to the versions
in force (it reads the current version and adds four statements to the
policy and one to the boundary, so nothing already there is lost):

```bash
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
R=eu-central-1

cat > chat-cfn-exec.json <<EOF
[
  {"Sid": "ChatFunction", "Effect": "Allow", "Action": "lambda:*",
   "Resource": ["arn:aws:lambda:${R}:${ACCOUNT}:function:scout-*",
                "arn:aws:lambda:${R}:${ACCOUNT}:function:scout-*:*"]},
  {"Sid": "ChatLogGroup", "Effect": "Allow", "Action": "logs:*",
   "Resource": ["arn:aws:logs:${R}:${ACCOUNT}:log-group:/aws/lambda/scout-*",
                "arn:aws:logs:${R}:${ACCOUNT}:log-group:/aws/lambda/scout-*:*"]},
  {"Sid": "ChatDescribeLogGroups", "Effect": "Allow", "Action": "logs:DescribeLogGroups",
   "Resource": "arn:aws:logs:${R}:${ACCOUNT}:log-group:*"},
  {"Sid": "ChatCodeFromCdkAssets", "Effect": "Allow", "Action": "s3:GetObject",
   "Resource": "arn:aws:s3:::cdk-hnb659fds-assets-${ACCOUNT}-${R}/*"}
]
EOF
cat > chat-boundary.json <<EOF
[
  {"Sid": "ChatLambdaLogs", "Effect": "Allow",
   "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
   "Resource": "arn:aws:logs:${R}:${ACCOUNT}:log-group:/aws/lambda/scout-*:*"}
]
EOF

add_statements() {  # policy name, file of statements; replaces statements of the same Sid
  ARN="arn:aws:iam::${ACCOUNT}:policy/$1"
  V=$(aws iam get-policy --policy-arn "$ARN" --query Policy.DefaultVersionId --output text)
  aws iam get-policy-version --policy-arn "$ARN" --version-id "$V" \
    --query PolicyVersion.Document > current.json
  python3 - "$2" > new.json <<'PY'
import json, sys
doc = json.load(open("current.json"))
add = json.load(open(sys.argv[1]))
sids = {s["Sid"] for s in add}
doc["Statement"] = [s for s in doc["Statement"] if s.get("Sid") not in sids] + add
print(json.dumps(doc))
PY
  aws iam create-policy-version --policy-arn "$ARN" --policy-document file://new.json \
    --set-as-default
}
add_statements scout-cfn-exec chat-cfn-exec.json
add_statements scout-boundary chat-boundary.json

# Reserved concurrency needs room in the account: a new account's limit can be 10,
# with 10 kept unreserved. This account's is 10 (26 October), so the stack has no
# reserved concurrency until Service Quotas (AWS Lambda, Concurrent executions)
# raises it; then restore reserved_concurrent_executions=2 (the TODO in chat_stack.py).
aws lambda get-account-settings --query 'AccountLimit.[ConcurrentExecutions,UnreservedConcurrentExecutions]'
```

A policy keeps at most five versions; if `create-policy-version` says so,
delete the oldest one that is not the default first (`aws iam
list-policy-versions`, then `aws iam delete-policy-version --version-id
v1`). `lambda:*` is limited to functions named `scout-*`, whose roles must
carry `scout-boundary` like every other project role; the CDK staging bucket
is read because Lambda fetches the function's code from it with the
deployer's rights.

### Concurrency: the account limit of 10 is the cap for now

The account's Lambda concurrency limit is 10, and AWS keeps 10 unreserved, so
reserving 2 for `scout-chat` fails. Until the quota increase is granted the
stack sets no reserved concurrency: the chat shares the account's 10, and
that limit of 10 is its concurrency cap (the caps on visitors and money
above are unchanged; a burst of 10 turns at once can pass the daily budget by
a few tens of cents before the next check). Once Service Quotas raises the
limit, restore `reserved_concurrent_executions=2` (the one line TODO in
`infra/scout_infra/chat_stack.py` and the test in
`infra/tests/test_chat_stack.py`) and redeploy.

### Deploying and redeploying (a session, with the owner's yes)

The Claude key is passed as the NoEcho stack parameter `AnthropicApiKey` from
the deploying shell's `ANTHROPIC_API_KEY`: it is never written to a file,
never in the synthesized template or `cdk.out`, and CloudFormation shows it
masked. In a cloud session the variable comes from the environment's
settings (never pasted into a chat).

```bash
cd infra && CDK_DEFAULT_ACCOUNT=$(uv run python -c \
  "import boto3; print(boto3.client('sts').get_caller_identity()['Account'])") \
  npx aws-cdk@2 deploy ScoutChat --require-approval never \
  --parameters ScoutChat:AnthropicApiKey="$ANTHROPIC_API_KEY"
```

The output `ScoutChat.ChatUrl` is the Function URL. Put it into
`configs/backfill.ultra.yaml` (`chat.url`) and commit: the next weekly run
(or a run started by hand) publishes the page with the Ask tab switched on.
Later deploys (a code change) need no `--parameters`: CDK keeps the previous
value. To rotate the key, deploy again with the new key in
`ANTHROPIC_API_KEY`. `cd infra && uv run pytest` checks the stack (one
function, the NoEcho key, CORS for the site only, the role's five actions,
a private retained bucket).

### Cost

At about 0.02 to 0.04 USD a question (two or three model calls, the tools and
the system prompt read from the cache after the first), the visitors' caps
hold the chat under 1 USD a day and 5 USD a month. The Lambda (512 MB, a few
seconds a question), its S3 requests and its logs stay inside AWS's free
tier at this scale.

### The evaluation

```bash
uv run scout --config configs/backfill.ultra.yaml chat-eval --dry-run   # build the set, check the tools
uv run scout --config configs/backfill.ultra.yaml chat-eval             # against chat.url (ANTHROPIC_API_KEY)
uv run scout --config configs/backfill.ultra.yaml chat-eval --local     # the same chat in this process
```

`src/scout/chat_eval.py` builds 50 questions from the published files (read
from `chat.site_url`): 30 whose answers it computes with pandas and numpy,
independently of the tools, 10 the tables cannot answer, and 10 attempts to
pull the system prompt or go off topic. It scores exact names and numbers,
the fixed "not in the data" sentence and the fixed refusal, and puts each
failure down to the model's answer, the model's tool use, or the tools (by
running the ideal tool call itself). Results: `reports/chat_eval.md` and
`reports/chat_eval.json`. A run costs about 1 to 2 USD.

## Retraining (manual, never in the weekly run)

Retrain when a new season of data arrives (the 6 November pull), when a new
value source replaces Transfermarkt, or after an adapter repair. On a machine
with about 10 GB of memory:

```bash
C="uv run scout --config configs/backfill.ultra.yaml"
$C sync down --only processed && $C sync down --only transfermarkt
$C features            # full rebuild, about 10 minutes
$C backtest            # fair value; stores fair_value_fold_<season>
$C moves               # big move; stores big_move_24m
$C scouting            # ability fit, elite, step up, riser; stores them
$C embed --train       # similar players: tests, refit, stores embedding.json (about 3 minutes)
$C forecast history    # when a new season can be scored
$C lists && $C site
$C sync up --only processed   # the owner's yes first (rule 8)
$C publish --role scout-site-publisher
```

Each stored model has a `.json` beside it with its columns, what it was
trained on and the date. If the feature columns change, `scout score` stops
with a message to retrain rather than score with a model that expects other
columns.

## The free tier update (for prompt 4)

The run's step 3 is a placeholder with a fixed contract:

* it runs only when `configs/update.free.yaml` exists, and then calls
  `scout --config configs/update.free.yaml update`; if the config exists but
  the command does not, the run fails and says so;
* the command must read and write the raw cache in S3, so no request is ever
  made twice (rule 4; the runner starts empty every week), and write back the
  processed partitions it changes (and the Elo tables, if fixtures change);
* `scout-weekly` then needs read and write on `raw/api_football/` and write
  on those partitions. That is beyond the rule 8 exception for this job and
  is the owner's decision.

`features --current` also rebuilds the season so far block of both tables
(prompt 16) from `processed/current/`, which the run pulls with the rest of
`processed/`. The run never calls `scout current` (it spends about 2,000
requests: squads and transfer lists of 830 clubs, the finished matches), so
the club today and the season so far stay as of the last pull until a
session runs `scout current` and syncs `raw/` and `processed/` up. On the
free tier (100 a day from 7 November) a full pull takes about three weeks;
a week of new matches alone (37 fixture lists and about 50 new batches of
20) fits in one day.

A new season in the season stats makes `features --current` build that
season. Its feature date (30 June 2027) lies more than a year after the last
Transfermarkt valuation (12 June 2026), so its rows would have no market
value and no candidates: an in progress season needs its own feature date
rule and a new value source first (CLAUDE.md, open questions).

## Undoing

```bash
aws iam list-access-keys --user-name scout-weekly   # then delete-access-key for each
aws iam delete-user-policy --user-name scout-weekly --policy-name scout-weekly
aws iam delete-user --user-name scout-weekly
cd infra && CDK_DEFAULT_ACCOUNT=<account id> npx aws-cdk@2 destroy ScoutSite
```

`destroy` keeps the site bucket (it is retained); empty and delete it by hand
to take the page down.

The chat: `cd infra && CDK_DEFAULT_ACCOUNT=<account id> npx aws-cdk@2 destroy
ScoutChat` removes the function and its URL (the page's Ask tab then says
the chat cannot be reached; set `chat.url` back to `null` to hide it) and
keeps the chat bucket and the log group, which are retained.
