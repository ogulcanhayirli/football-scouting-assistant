# Registered forecast of 2026-10-21

The two user facing tables exactly as `scout lists` wrote them on 2026-10-21, frozen so they can be scored when the outcomes are known. Never edit these files; a new forecast gets a new dated folder.

* Season: 2025 (2025/26, or the calendar year for calendar year leagues).
* Feature date (the day each row describes the player): 2026-01-01 for Allsvenskan, Brasileirão Série A, Colombian Primera A, Eliteserien, J1 League, K League 1, Liga Profesional Argentina, Major League Soccer; 2026-06-30 for 2. Bundesliga, Austrian Bundesliga, Belgian Pro League, Bundesliga, Championship, Croatian HNL, Czech First League, Danish Superliga, Eerste Divisie, Ekstraklasa, Eredivisie, Greek Super League, La Liga, Liga MX, Ligue 1, Ligue 2, Premier League, Primeira Liga, Russian Premier League, Saudi Pro League, Scottish Premiership, Segunda División, Serbian SuperLiga, Serie A, Serie B, Swiss Super League, Süper Lig, Ukrainian Premier League. No row uses anything dated after it.
* Market values: the Transfermarkt snapshot, whose valuations end on 2026-06-12 (the latest value in these tables is dated 2026-06-11).
* Rows: 1,830 candidates, 1,602 bargains.
* Revised: 2026-10-21, 2026-10-21 (see Revisions); the date and the rows are those of the original.
* Due: 2 seasons on, at the feature dates of season 2027 (two years after the dates above); an interim score is possible after one season.
* Score it with `uv run scout --config configs/backfill.ultra.yaml forecast score forecasts/2026-10-21 --as-of <later feature date>`.

## What each column promises

Every chance is a historical rate: of the players that past forecasts, rebuilt for each season with the models and buckets of that time and scored two seasons on, put in the same bucket, the share who made it (the realized rate, since prompt 14b). A well calibrated forecast sees about that share of this season's players in the bucket make it too.

* `elite_chance_pct`: that share of the bucket's players reach the 100 most valuable players within two seasons (elite pool only: ranked #101 to #1,000, 23 or under).
* `fall_back_risk_pct`: that share of the elite pool players with the same flag count (0, 1, 2 or more) have a value rank at least twice as bad two seasons on.
* `step_up_chance_pct`: that share of the bucket's players (ranked #101 to #2,000) see their value rank become at least three times better, inside the top 500, within two seasons.
* `big_move_chance_pct`: that share of the bucket's players (not at a top 20 club) join a top 20 club by squad value within 24 months, a return to a former club not counted.
* `rising`: the riser model's top 50; the riser backtest saw about 9% of such a top 50 rise, against a base rate under 2%.
* `bargains`: the top 50 of each list (`list_rank` 1 to 50) gain market value over 24 months more often than the rest of the list.
* `flags`, `explanation`, values and ability are descriptions, not promises.

## Columns

| table      | column                  | meaning                                                                                                                                                                                                                                                                                                           |
|:-----------|:------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| candidates | season                  | The season the row describes (2025 is 2025/26, or the 2025 calendar year).                                                                                                                                                                                                                                        |
| candidates | player_id               | The player's API Football id, for joining with other tables.                                                                                                                                                                                                                                                      |
| candidates | player                  | The player's name as API Football writes it.                                                                                                                                                                                                                                                                      |
| candidates | age                     | His age in years at the feature date, the day the row describes him.                                                                                                                                                                                                                                              |
| candidates | position                | His Transfermarkt position, such as Centre-Forward or Left-Back.                                                                                                                                                                                                                                                  |
| candidates | club                    | The club he played most league minutes for that season.                                                                                                                                                                                                                                                           |
| candidates | league                  | The league of that club.                                                                                                                                                                                                                                                                                          |
| candidates | value_rank              | Where his market value ranks among all players we track (1 is the dearest); blank when he has no market value.                                                                                                                                                                                                    |
| candidates | elite_decile            | Where the elite model puts him among this season's elite pool, from 1 (least likely) to 10 (most likely); blank outside the pool.                                                                                                                                                                                 |
| candidates | elite_chance_pct        | Of the players that past forecasts of the last five scored seasons, rebuilt with the models of their time, showed in the same elite decile, the share who became one of the 100 most valuable players within two seasons, rounded to 5; blank outside the elite pool (ranked 101 to 1,000 and 23 or under).       |
| candidates | elite_chance_players    | How many players of those past forecasts, with a known outcome, that elite chance is based on.                                                                                                                                                                                                                    |
| candidates | step_up_decile          | Where the step up model puts him among this season's candidates, from 1 to 10.                                                                                                                                                                                                                                    |
| candidates | step_up_chance_pct      | Of the players ranked 101 to 2,000 that past forecasts of the last five scored seasons showed in the same step up decile, the share whose value rank became at least three times better, inside the top 500, within two seasons, rounded to 5.                                                                    |
| candidates | step_up_chance_players  | How many players of those past forecasts, with a known outcome, that step up chance is based on.                                                                                                                                                                                                                  |
| candidates | fall_back_risk_pct      | Of the elite pool players that past forecasts of the last five scored seasons showed with the same number of flags, the share whose value rank was at least twice as bad two seasons later, rounded to 5; blank outside the elite pool, where it was never measured.                                              |
| candidates | fall_back_risk_players  | How many players of those past forecasts, with a known outcome, that fall back risk is based on.                                                                                                                                                                                                                  |
| candidates | flag_count              | How many warning flags he raises: low minutes, injured, value falling.                                                                                                                                                                                                                                            |
| candidates | flags                   | The raised flags, each with his own number, or none.                                                                                                                                                                                                                                                              |
| candidates | explanation             | The 3 stats where he stands furthest from regulars of his position and age band this season, each with his number and where it ranks (top means better, even for stats where less is better); a neutral sentence when nothing is in the top or bottom 10%. It reads the same numbers as the flags.                |
| candidates | rising                  | True when the riser model has him in this season's top 50: likely to play clearly better in each of the next two seasons and end at the 70th percentile of his position or better (players with at least 50% of team minutes). Risers valued below #2,000 are added as rows of their own, with the chances blank. |
| candidates | big_move_decile         | Where the big move model puts him among this season's candidates, from 1 to 10; blank when he already plays for a top 20 club.                                                                                                                                                                                    |
| candidates | big_move_chance_pct     | Of the players ranked 101 to 2,000 that past forecasts of the last five scored seasons showed in the same big move decile, the share who joined a top 20 club (by squad value) within 24 months, rounded to 5.                                                                                                    |
| candidates | big_move_chance_players | How many players of those past forecasts, with a known outcome, that big move chance is based on.                                                                                                                                                                                                                 |
| candidates | value_eur               | His Transfermarkt market value at the feature date, in euros.                                                                                                                                                                                                                                                     |
| candidates | fair_value_eur          | What the fair value model, trained on earlier seasons only, says a player with his record is worth, in euros.                                                                                                                                                                                                     |
| candidates | ability_pct             | His ability score as a percentile among regulars of his position that season (100 is the best), from per 90 stats of the last two seasons.                                                                                                                                                                        |
| bargains   | season                  | The season the row describes.                                                                                                                                                                                                                                                                                     |
| bargains   | list                    | prime (29 and under) or experienced (30 to 33).                                                                                                                                                                                                                                                                   |
| bargains   | list_rank               | His place on that list, 1 being the cheapest against the fair value.                                                                                                                                                                                                                                              |
| bargains   | player_id               | The player's API Football id.                                                                                                                                                                                                                                                                                     |
| bargains   | player                  | The player's name as API Football writes it.                                                                                                                                                                                                                                                                      |
| bargains   | age                     | His age in years at the feature date.                                                                                                                                                                                                                                                                             |
| bargains   | position                | His Transfermarkt position.                                                                                                                                                                                                                                                                                       |
| bargains   | club                    | The club he played most league minutes for that season.                                                                                                                                                                                                                                                           |
| bargains   | league                  | The league of that club.                                                                                                                                                                                                                                                                                          |
| bargains   | ability_pct             | His ability percentile within his position (70 or better to be listed).                                                                                                                                                                                                                                           |
| bargains   | minutes_share           | The share of his club's league minutes he played (half or more to be listed).                                                                                                                                                                                                                                     |
| bargains   | value_eur               | His Transfermarkt market value at the feature date, in euros.                                                                                                                                                                                                                                                     |
| bargains   | fair_value_eur          | What the fair value model, trained on earlier seasons only, says he is worth, in euros.                                                                                                                                                                                                                           |
| bargains   | value_to_fair           | Market value divided by fair value: 0.40 means the market asks 40% of the model's price.                                                                                                                                                                                                                          |
| bargains   | value_to_fair_pct       | That ratio as a percentile among listed players of his position group (1 is the cheapest); the lists are sorted by it.                                                                                                                                                                                            |
| bargains   | contract_expiry         | When his contract ends, from the Transfermarkt snapshot, shown only when it ends after the snapshot date.                                                                                                                                                                                                         |
| bargains   | flag_count              | How many warning flags he raises: low minutes, injured, value falling.                                                                                                                                                                                                                                            |
| bargains   | flags                   | The raised flags, each with his own number, or none.                                                                                                                                                                                                                                                              |
| bargains   | explanation             | The 3 stats where he stands furthest from regulars of his position and age band this season, each with his number and where it ranks (top means better, even for stats where less is better); a neutral sentence when nothing is in the top or bottom 10%. It reads the same numbers as the flags.                |
| bargains   | on_loan                 | True when he is on loan: the parent club decides the sale and the price.                                                                                                                                                                                                                                          |
| bargains   | stale_value             | True when his market value is dated more than 180 days before the feature date, so the cheap price may be out of date.                                                                                                                                                                                            |
| bargains   | value_date              | The date of his market value.                                                                                                                                                                                                                                                                                     |

## Revisions

### Revision 1, 2026-10-21

Prompt 14b, the owner's calibration fix. Scoring the rebuilt forecasts of 2015 to 2023 showed the elite chance too optimistic at the top (players shown 40% reached the top 100 30.5% of the time), because each shown chance was the in sample rate: the backtest rows of every test season pooled. Every chance (elite by decile, step up and big move by decile, fall back by flag count) is now the realized rate of those rebuilt forecasts, built with the models and buckets of their time and scored two seasons on (seasons 2015 to 2023, all scored by the 30 June 2026 feature date), so the table promises what a user of past tables actually saw happen. The 30 player floor (on known outcomes) and the rounding to 5% stay; the count is now the number of past forecast players with a known outcome. Every bucket's realized rate rises with its neighbours. Players, deciles, flags, order and the bargains table are unchanged.

Changed: candidates.csv, dictionary.csv. The original is in git at commit e012805.

Every bucket of this forecast, old against new: the shown share and the number of past players behind it, and how many of this season's players are in it.

| promise   | bucket          |   players now | old                | new                |
|:----------|:----------------|--------------:|:-------------------|:-------------------|
| elite     | decile 1        |            32 | under 5% (n=287)   | under 5% (n=240)   |
| elite     | decile 2        |            33 | under 5% (n=294)   | under 5% (n=244)   |
| elite     | decile 3        |            33 | 5% (n=292)         | 5% (n=243)         |
| elite     | decile 4        |            33 | 5% (n=293)         | 5% (n=246)         |
| elite     | decile 5        |            33 | 5% (n=295)         | 5% (n=244)         |
| elite     | decile 6        |            33 | 10% (n=290)        | 10% (n=242)        |
| elite     | decile 7        |            33 | 15% (n=292)        | 15% (n=245)        |
| elite     | decile 8        |            33 | 20% (n=293)        | 15% (n=244)        |
| elite     | decile 9        |            33 | 25% (n=293)        | 20% (n=243)        |
| elite     | decile 10       |            33 | 40% (n=296)        | 35% (n=249)        |
| step up   | decile 1        |           179 | under 5% (n=1,972) | under 5% (n=1,543) |
| step up   | decile 2        |           179 | under 5% (n=1,978) | under 5% (n=1,561) |
| step up   | decile 3        |           179 | under 5% (n=1,974) | under 5% (n=1,558) |
| step up   | decile 4        |           179 | under 5% (n=1,978) | under 5% (n=1,570) |
| step up   | decile 5        |           180 | 5% (n=1,977)       | under 5% (n=1,573) |
| step up   | decile 6        |           179 | 5% (n=1,976)       | 5% (n=1,573)       |
| step up   | decile 7        |           179 | 5% (n=1,975)       | 5% (n=1,575)       |
| step up   | decile 8        |           179 | 10% (n=1,977)      | 10% (n=1,576)      |
| step up   | decile 9        |           179 | 15% (n=1,975)      | 10% (n=1,575)      |
| step up   | decile 10       |           180 | 20% (n=1,983)      | 20% (n=1,583)      |
| big move  | decile 1        |           144 | under 5% (n=1,659) | under 5% (n=1,323) |
| big move  | decile 2        |           145 | under 5% (n=1,666) | under 5% (n=1,327) |
| big move  | decile 3        |           145 | under 5% (n=1,664) | under 5% (n=1,327) |
| big move  | decile 4        |           145 | 5% (n=1,664)       | 5% (n=1,326)       |
| big move  | decile 5        |           145 | 5% (n=1,668)       | 5% (n=1,328)       |
| big move  | decile 6        |           145 | 5% (n=1,663)       | 5% (n=1,326)       |
| big move  | decile 7        |           145 | 5% (n=1,663)       | 5% (n=1,326)       |
| big move  | decile 8        |           145 | 10% (n=1,665)      | 10% (n=1,328)      |
| big move  | decile 9        |           145 | 15% (n=1,665)      | 15% (n=1,327)      |
| big move  | decile 10       |           145 | 25% (n=1,670)      | 25% (n=1,331)      |
| fall back | flags 0         |           146 | 15% (n=1,318)      | 15% (n=1,298)      |
| fall back | flags 1         |           106 | 20% (n=788)        | 20% (n=720)        |
| fall back | flags 2 or more |            77 | 25% (n=405)        | 25% (n=384)        |

### Revision 2, 2026-10-21

Prompt 14c, the owner's decision on the prompt 14b report. Pooling the realized rates of every scored season from 2015 still leaned on 2015 to 2017: the top elite decile made it 27.6% to 32.1% in each season 2020 to 2023 against 35% shown, and the top fall back bucket showed 35% against 23.4% realized. Every displayed chance (elite by decile, step up and big move by decile, fall back by flag count) now uses a rolling window of the last five scored seasons, so this season 2025 table reads the forecasts of 2019 to 2023 instead of 2015 to 2023. The 30 player floor on known outcomes, the rounding to 5% and the count stay; a bucket under the floor would merge with its neighbour, and none did. The candidates are now sorted by elite decile, then step up decile, instead of by the unrounded rate: over 2019 to 2023 elite decile 1 (2.9%) came out above decile 2 (0.7%), which would have moved players; while the rates rise with the decile, as in revision 1, both give the same order. Players, order, deciles, flags, explanations, values and the bargains table are unchanged.

Changed: candidates.csv, dictionary.csv. The original is in git at commit 38a8a4b.

Every bucket of this forecast, old against new: the shown share and the number of past players behind it, and how many of this season's players are in it.

| promise   | bucket          |   players now | old                | new              |
|:----------|:----------------|--------------:|:-------------------|:-----------------|
| elite     | decile 1        |            32 | under 5% (n=240)   | 5% (n=137)       |
| elite     | decile 2        |            33 | under 5% (n=244)   | under 5% (n=140) |
| elite     | decile 3        |            33 | 5% (n=243)         | 5% (n=138)       |
| elite     | decile 4        |            33 | 5% (n=246)         | 5% (n=141)       |
| elite     | decile 5        |            33 | 5% (n=244)         | 5% (n=139)       |
| elite     | decile 6        |            33 | 10% (n=242)        | 10% (n=139)      |
| elite     | decile 7        |            33 | 15% (n=245)        | 15% (n=140)      |
| elite     | decile 8        |            33 | 15% (n=244)        | 15% (n=139)      |
| elite     | decile 9        |            33 | 20% (n=243)        | 20% (n=139)      |
| elite     | decile 10       |            33 | 35% (n=249)        | 30% (n=142)      |
| step up   | decile 1        |           179 | under 5% (n=1,543) | under 5% (n=865) |
| step up   | decile 2        |           179 | under 5% (n=1,561) | under 5% (n=869) |
| step up   | decile 3        |           179 | under 5% (n=1,558) | under 5% (n=868) |
| step up   | decile 4        |           179 | under 5% (n=1,570) | under 5% (n=876) |
| step up   | decile 5        |           180 | under 5% (n=1,573) | under 5% (n=875) |
| step up   | decile 6        |           179 | 5% (n=1,573)       | 5% (n=877)       |
| step up   | decile 7        |           179 | 5% (n=1,575)       | 5% (n=876)       |
| step up   | decile 8        |           179 | 10% (n=1,576)      | 10% (n=877)      |
| step up   | decile 9        |           179 | 10% (n=1,575)      | 10% (n=875)      |
| step up   | decile 10       |           180 | 20% (n=1,583)      | 25% (n=878)      |
| big move  | decile 1        |           144 | under 5% (n=1,323) | under 5% (n=734) |
| big move  | decile 2        |           145 | under 5% (n=1,327) | under 5% (n=736) |
| big move  | decile 3        |           145 | under 5% (n=1,327) | under 5% (n=736) |
| big move  | decile 4        |           145 | 5% (n=1,326)       | 5% (n=736)       |
| big move  | decile 5        |           145 | 5% (n=1,328)       | 5% (n=737)       |
| big move  | decile 6        |           145 | 5% (n=1,326)       | 5% (n=736)       |
| big move  | decile 7        |           145 | 5% (n=1,326)       | 10% (n=736)      |
| big move  | decile 8        |           145 | 10% (n=1,328)      | 10% (n=737)      |
| big move  | decile 9        |           145 | 15% (n=1,327)      | 15% (n=736)      |
| big move  | decile 10       |           145 | 25% (n=1,331)      | 25% (n=739)      |
| fall back | flags 0         |           146 | 15% (n=1,298)      | 15% (n=691)      |
| fall back | flags 1         |           106 | 20% (n=720)        | 20% (n=418)      |
| fall back | flags 2 or more |            77 | 25% (n=384)        | 25% (n=260)      |
