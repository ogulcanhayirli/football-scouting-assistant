# Archetypes: players without data (prompt 18c)

One YAML file per player, describing him in the style vector's own
dimensions so that `scout archetypes` can put him through the same level
free space as today's players and list the nearest of them. Every result
carries the label "archetype from the record, not measured".

`legends/` holds the players the page shows; `calibration/` holds six
players drafted the same way whose seasons we can measure, to test how far
a draft from the record lands from the measurement (`reports/archetypes.md`).

**Every file was drafted by a Claude Code session on 25 October 2026 from
the historical record and awaits the owner's review.** The session did not
watch matches and did not open the listed sources while drafting; it drew on
what they report. Edit any number or line, set `reviewed: true` with your
name and date, and rebuild:

```bash
C="uv run scout --config configs/backfill.ultra.yaml"
$C archetypes   # reloads every file, checks it, rebuilds the vectors, neighbours and report
$C site         # the page
```

## Fields

| Field | Meaning |
|---|---|
| `name` | The player. |
| `position_group` | The group he is compared in: `goalkeeper`, `centre_back`, `full_back`, `defensive_mid`, `central_mid`, `attacking_mid`, `winger`, `centre_forward`. |
| `era` | The seasons the traits describe. |
| `measured` | Always `false`: an archetype is drawn from the record. |
| `drafted` | Who drafted it and when. |
| `reviewed` | `false` until the owner has reviewed it; then `true` with `reviewed_by` and `reviewed_on`. |
| `traits` | Each of the 18 measures below: `percentile` (0 to 100) and `why` (one line). |
| `sources` | What the record rests on; at least one. |
| `calibration` | Calibration files only: the measured season (`source` `api_football` or `statsbomb`, `player_id`, `season`). |

## Traits

A percentile places the player among the regular players (450 league
minutes or more) of his position group **in his own seasons**, before the
level is taken out: 50 is a typical regular of that group, 90 the top tenth.
Each season is standardised on its own, so a percentile is against his
contemporaries, not against today's players. A dominant team's players
have more of the ball, so their passing and shooting volumes are high and
their defensive volumes low; a percentile describes what he did, not how
good he was at it. The level is taken out afterwards with the same curves
our own players go through, so what is compared is how he played against
players of his level.

| Trait | What it counts |
|---|---|
| `goals_per90` | Goals per 90 minutes (no own goals). |
| `assists_per90` | Assists per 90. |
| `shots_total_per90` | Shots per 90, on target or not. |
| `passes_key_per90` | Passes that lead to a shot, per 90. |
| `passes_total_per90` | Passes per 90 (throw ins left out). |
| `dribbles_success_per90` | Dribbles past an opponent per 90. |
| `dribble_success_rate` | Share of attempted dribbles that succeed. |
| `duels_won_per90` | Duels won per 90 (tackles, aerials, 50/50s, dribbles, fouls won). |
| `duel_win_rate` | Share of duels won. |
| `tackles_total_per90` | Tackles per 90. |
| `interceptions_per90` | Interceptions per 90. |
| `blocks_per90` | Shots blocked per 90. |
| `fouls_drawn_per90` | Fouls suffered per 90. |
| `saves_per90` | Saves per 90; an outfield player makes none, so 50 (the measure has no spread outside goal). |
| `share_shots` | His share of the team's shots while he is on the pitch (share of shots over share of minutes). |
| `share_dribbles` | The same for dribbles. |
| `share_passes` | The same for passes: how much the team's passing runs through him. |
| `share_duels` | The same for duels. |
