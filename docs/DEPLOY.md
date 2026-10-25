# The weekly run and the page

Decision 9: no Lambda, API Gateway, DynamoDB or Bedrock agent. Once a week a
GitHub Actions run rebuilds the current season's two tables with the stored
models and publishes them, with one static HTML page, to a public S3 website
bucket. The data bucket stays private.

```
 GitHub Actions, .github/workflows/weekly.yml (Mondays 05:17 UTC, or Run workflow)
   1. check the secrets and variables      fails naming the one that is missing
   2. scout sync down processed, transfermarkt   data bucket, read only
   3. free tier update                     skipped until configs/update.free.yaml exists
   4. scout features --current             the newest season only, earlier ones as stored
   5. scout score                          stored models (processed/models/stored/), no training
   6. scout embed                          stored embedding: vectors, today's ten nearest each
   7. scout lists, scout site              the two tables, explanations, index.html, similar.json
   8. scout publish                        processed/lists/ in the data bucket, the site bucket
                                                    │
 S3 football-scout-data-<owner> (private)            ▼
   processed/ ── read ──►  runner  ── write ──► processed/lists/
   raw/transfermarkt/ ──►           └─ write ──► S3 scout-site-<account> (public read, website)
                                                     index.html, data/*.csv, data/*.parquet,
                                                     data/similar.json,
                                                     statsbomb-logo.png (with the greats)
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
and write only `processed/lists/` and the site bucket. It cannot delete
anything, and it cannot touch `raw/api_football/`, `reports/` or any other
bucket.

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

### 3. GitHub secrets and variables

In the repository: Settings, Secrets and variables, Actions.

| Kind | Name | Value |
|---|---|---|
| Secret | `AWS_ACCESS_KEY_ID` | `AccessKeyId` of `scout-weekly` |
| Secret | `AWS_SECRET_ACCESS_KEY` | `SecretAccessKey` of `scout-weekly` |
| Secret | `API_FOOTBALL_KEY` | the API Football key (used by the free tier update only) |
| Variable | `SCOUT_S3_BUCKET` | `football-scout-data-ogulcanhayirli` |
| Variable | `SCOUT_SITE_BUCKET` | `scout-site-098499087060` |

The bucket names are variables, not secrets, so the logs can show the page
address. The first step of the run fails with the name of any secret or
variable that is missing.

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
| `lists` | 22 |
| `site` | 2 |
| `publish` (dry run) | 2 |
| **total** | **254** (271 with `embed`) |

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
