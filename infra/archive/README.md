# Archived stacks

Retired by decision 9 (prompt 15, 21 October 2026): the serving stack is now
a weekly GitHub Actions run that publishes the two tables and one static page
to S3 (`.github/workflows/weekly.yml`, `docs/DEPLOY.md`). Nothing in the
weekly run or the pipeline imports these stacks, and their tests are no
longer run in CI.

| Stack | What it held | Why it is retired |
|---|---|---|
| `ScoutData` | Glue tables over the feature Parquet, the `scout` Athena workgroup, the read only `scout-analyst` role | Built to query the features with SQL; the lists are built locally and the page needs no query engine. Its Glue schema was regenerated on 18 October but never redeployed. |
| `ScoutFeatureStore` | SageMaker Feature Store group, offline store (Iceberg) | A mirror kept to compare point in time answers with ours (D15); they matched, and it predates the context features. |
| `ScoutTraining` | `scout-sagemaker-training` and `scout-ml-runner` roles for training and tuning jobs | Tuning ran once, for the retired growth model; every model now trains locally in minutes (`scout backtest`, `moves`, `scouting`). |

The code is kept, not deleted, because **the three stacks are still deployed**
in eu-central-1. `scout athena`, `scout featurestore`, `scout train` and
`scout tune` keep working against them until they are removed. Removing them
is the owner's call (rule 8): from this folder,

```bash
CDK_DEFAULT_ACCOUNT=<account id> npx aws-cdk@2 destroy ScoutTraining ScoutFeatureStore ScoutData
```

`destroy` keeps the S3 buckets the stacks marked to retain; empty and delete
those by hand if wanted. The archived tests (`tests/`) still describe the
stacks but are not collected: `infra/` runs `tests/` only, and the two that
lived in `tests/unit` (`test_glue_schema.py`, `test_infra_config.py`) need
the main package.
