# How it was built

I am the owner: a football fan with a brief, not a professional scout and
not the one who typed the code. Between 7 and 29 October 2026 I wrote the
brief, made the decisions and reviewed the results. Coding agents (Claude
Code cloud sessions) wrote the code, one numbered prompt at a time.

## The way of working

* **A decisions log first.** CLAUDE.md holds the rules every change must
  follow (point in time features, season by season backtests, a baseline
  next to every model, a raw cache that is never fetched twice, no secrets
  in code, no data in git) and my numbered decisions 1 to 12. Each prompt
  was built against that log, and a prompt that changed a decision wrote
  the change back into it.
* **Named tests with every step.** Each prompt shipped with tests named
  for what they prove (for example a test that no feature row uses a date
  after its feature date, or that the chat's tools agree with pandas on
  every computed question). CI runs lint, the unit tests, the data
  contracts and the CDK tests on every push to `main`. 621 unit tests pass
  today.
* **A go before any spend or deploy.** Uploads to S3, stack deploys, paid
  API runs and manual weekly runs waited for me to type go. The one
  standing exception is the weekly run's own publish.
* **Numbers next to their baseline, failures written down.** MODELS.md
  records every result with its baseline, including the ones that lost.
  CLAUDE.md (Open questions) records what is still unresolved.

## The prompts

The first sessions, 7 to 18 October, were not numbered in the git log.
Prompts 13 and 17 do not appear in the git log. Prompt 4, the free tier
update, is planned for 6 November and not built.

| Prompt | Date | What it built | What it proved or disproved |
|---|---|---|---|
| early | 7 Oct | Raw cache, quota guard, API Football client, Transfermarkt adapter, data contracts, CI, backfill, player matching, feasibility check | 99% of regular players from 2015 have a market value at 30 June, so season end works as the feature date (D9); the season endpoint undercounts Süper Lig minutes, so minutes come from match data (D12) |
| early | 8 to 9 Oct | Per match stats, point in time features, Glue and Athena, Feature Store, first backtest, SageMaker training and tuning | Feature Store's point in time answers matched ours on every player checked (D15); tuning was run once, for a growth model later retired |
| early | 11 to 13 Oct | 17 leagues, transfers, cup, team, injury and transfer features, big move model, first scouting pass | Big move: AUC 0.907 against 0.780 for its baseline on these 17 leagues |
| early | 16 to 18 Oct | My decisions 1 to 9, CLAUDE.md, team Elo from our own fixtures, 20 more leagues, sidelined periods, coaches, match events | Elo beats the home rate baseline (Brier 0.161 against 0.180) |
| early | 19 Oct | Ability score version 2 (decision 8), bargains, fall back risk and flags, risers | Version 2 is far more stable year to year (0.858 against 0.503) and predicts as well, so it was adopted; the riser list beat its base rate, so it was kept |
| 12 | 19 Oct | Fixes after my review, then two tables only | The growth model barely beat its baseline and was retired; the ability flag missed my 1.3 rule and left the flag count |
| 14 | 21 Oct | One line explanations, the first registered forecast | |
| 14b | 21 Oct | Every chance shown as the realized rate of rebuilt past forecasts | The old top elite chance said 40% where 30.5% happened |
| 14c | 21 Oct | A window of the last five scored seasons | The top decile now shows 30%; the test can barely tell the window from pooling every season |
| 15 | 21 Oct | Weekly GitHub Actions run, stored models, the static page; old stacks archived | The weekly path rebuilt the registered tables byte for byte |
| 16 | 24 Oct | Club today, season so far, the January window forecast | |
| 18 | 24 Oct | Similar players: style vectors and a learned embedding | The embedding finds the same player 19.9% of the time against 13.2% for plain vectors; real replacements are only weakly similar (1.52% against 0.65% at random) |
| 18b | 25 Oct | The measured greats from StatsBomb open data | The two sources agree at 0.726 to 0.999 per player and match on most measures (tackles and interceptions fell under 0.7); a thin peer group alone drops the same player hit from 100% to 15.8% |
| 18c | 25 Oct | The peer group correction, archetypes for legends without data | The correction lifts the hit rate from 28.6% to 55.1%; blind drafts land within 15 points on only half the traits, so archetypes are labelled illustrative |
| 19 | 25 Oct | Written reports, a check in code, a weekly judge | The check worked, but 94% of reports failed on length |
| 19b | 25 Oct | Length handling | Run 9: 8 of 2,734 reports left on the rule text, for 0.74 USD |
| 19c | 28 Oct | The judge's verdicts sorted by hand, a typed check | The typed check catches 8 of the 9 real errors the old one missed |
| 20 | 26 Oct | The chat: one Lambda, four tools, caps | 28 of 30 computed, 10 of 10 unanswerable, 10 of 10 refused, 0.34 USD |
| 20b | 27 Oct | Both failures fixed, faster cold start, warm ping | 30 of 30; cold start 6.2 s at 512 MB, 3.0 s at 1,024 MB (measured with CPU held to Lambda's share) |
| 20c | 28 Oct | The same evaluation through the public Function URL | 30 of 30 again; the per visitor cap can be dodged by changing address |
| 20d | 28 Oct | The page opens on the chat; nicknames, legends, greats | |
| 20e | 28 Oct | A minimal page: the chat alone in view | |
| 20f | 28 Oct | Every player of the season, not only the two lists | 18,931 players, each with the code's reason for being on no list |
| 20g | 28 Oct | Decision 12: a bargain is priced below fair value | |
| 20h | 28 Oct | The 72 questions with the model, then shipped | 72 of 72 |
| 20i | 29 Oct | Plain text answers, stable tie breaks | 72 of 72 again |
| 21 | 29 Oct | This prompt: the repository made fit to open | The history holds no secret (gitleaks and a regex pass over every blob) |

## What went wrong and was corrected

* **The lists did not match the brief.** The first elite pool had no age
  limit, so a 20 year old and a 28 year old shared one fall back rate
  (37.7%). After my review (prompt 12) the pool became #101 to #1,000 and
  23 or under (17.2%), the elite chance moved from quintiles to deciles,
  the risers got a floor (without it the list leaned to goalkeepers and
  centre backs with low starting scores), experienced bargains stopped at
  33, the bargains' own flags gave way to one shared set, and the separate
  lists became two tables.
* **The chances were too confident.** The first tables showed in sample
  rates. Rebuilt past forecasts showed that 30.5% of players shown 40%
  reached the top 100. Prompts 14b and 14c replaced every chance with the
  realized rate and revised the registered forecast the same day, with
  both revisions recorded in its folder and the original kept in git.
* **The word bargain was wrong.** The bargains list ranked every eligible
  player by value against fair value, including the 72.7% of the prime
  list priced above it, so Mbappé (#550) and Haaland (#735) were on the
  "bargains" list and 352 written reports called such players a bargain.
  Decision 12 (prompt 20g) kept the ranking but reserved the word for
  players below fair value, on the page, in the chat and in a check on the
  reports.
* **The tie bug.** Two players with the same value and minutes could swap
  value ranks between builds (Beraldo and Matanović, both €22m). The swaps
  changed about 49 fact sheets a week, so the weekly run paid to rewrite
  their reports for nothing. Prompt 20i breaks such ties by player id.
* **The four minute outage.** The first redeploy of prompt 20h made the
  chat load a new file (`players.json`) before the live page had it. S3
  answered AccessDenied and the chat returned HTTP 503 from 19:36 to 19:40
  UTC on 28 October. The fix lets the chat start without an optional file.
* **The judge made its own mistakes.** Of the 24 reports the weekly judge
  called unfaithful in run 9, a hand sorting found 9 real errors, 8 judge
  mistakes and 7 the fact sheet left open. The judge now has to return the
  span and the field it contradicts, and code checks that span exists.
* **Smaller ones.** The first real report run cost 1.41 USD against an
  estimate of 0.70 to 1.10. Both chat failures in the first evaluation were
  ours, not the model's: the tool schema's field order stopped it adding an
  age after a league, and the leak guard blocked a right answer. 52
  bargains valued at 999,999.9999999997 were written as "€1000k". A live
  answer showed markdown asterisks on the page, so answers became plain
  text. 741 garbled names turned up where about ten were expected. One
  session downloaded 2.83 GB of StatsBomb data again before noticing it was
  already in the bucket.

## What I would check first as a reviewer

* `src/scout/models/backtest.py`: the folds, the gap and the leak check.
* `src/scout/contracts/schemas.py`: the point in time contract.
* `src/scout/candidates.py`: how a chance becomes a realized rate.
* `src/scout/reports_text.py`: the check every written report must pass.
* `src/scout/chat/` and `src/scout/chat_eval.py`: the chat and its 72
  questions.
