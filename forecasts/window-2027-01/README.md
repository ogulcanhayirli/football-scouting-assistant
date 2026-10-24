# Window forecast window-2027-01

Frozen on 2026-10-24. **Rule: joins a top 20 club by Elo or value rank between 1 January and 3 February 2027.**

Two lists, each scored against the rate of the pool it was chosen from:

* `big_move.csv`: the big move model's top 50 of season 2025 (50 players; pool 12,832, `pool_big_move.csv`): regular players not at a top 20 club by squad value at the feature date.
* `step_up.csv`: the step up model's top 50 among players ranked #101 to #2,000 by value (50 players; pool 1,413, `pool_step_up.csv`).

Both pools leave out players whose club at the freeze (the squads of the freeze day, else the club at forecast) is already one of the clubs in `top_clubs.csv`: the top 20 by Elo (rating after the latest league match, clubs that played in the 12 months before the freeze) and the top 20 by squad value in season 2025 (the big move model's clubs), 24 clubs in all.

A move counts when it is dated 1 January to 3 February 2027, goes to one of those clubs other than the club at the freeze, and is a transfer, a loan or a free move (a loan return is not a new move). A player with no move in that window is a miss; the transfers come from API Football's `/transfers` per club.

Score with `scout forecast window-score forecasts/window-2027-01` once a transfer snapshot after 3 February 2027 exists (`scout current`). The lists are refreshed and frozen again on 15 December 2026 (`scout forecast window-freeze --refreeze`); this freeze stays in git and in `window.json`.

Columns of the lists: `rank`, `player_id` (API Football), `player`, `age` (at the feature date), `position`, `club_at_forecast` (most league minutes in season 2025), `club_at_freeze` and `club_id_at_freeze`, `league`, `value_eur` (Transfermarkt, up to 12 June 2026), `score` (the model's score, not a probability).
