"""Does the StatsBomb mapping measure what API Football measures? (prompt 18b)

La Liga 2015/16 to 2020/21 exists in both sources (2015/16 in full,
Barcelona's matches after). A StatsBomb match is linked to its API Football
fixture by date, score and team names, and a player within a linked match
and side by shirt number with a name check. For the same player and match
each measure is compared: Pearson correlation and the mean difference
(StatsBomb minus API Football). A style measure whose counts correlate
below ``MIN_CORRELATION`` is left out of the StatsBomb vectors (read as the
position mean, as a missing measure is), unless the owner kept it (``KEPT``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from scout.matching import normalize_club, normalize_name

MIN_CORRELATION = 0.7
# Kept in the vectors although below the floor (owner, prompt 18c): tackles 0.625 and
# interceptions 0.675 by match, but the thin seasons' identity check found the same
# player twice as often with them (28.6% against 14.3%). The identity test decided.
KEPT = ("tackles_total_per90", "interceptions_per90")
CHECK_SEASONS = range(2015, 2021)  # 2015/16 to 2020/21
MIN_TEAM_SCORE = 60.0  # mean name similarity of the two sides of a linked match
MIN_NAME_SCORE = 60.0  # a shirt number match also needs a name this similar

# The match counts behind each style measure (scout.embeddings.STYLE_COLUMNS).
STYLE_COUNTS: dict[str, list[str]] = {
    "goals_per90": ["goals"],
    "assists_per90": ["assists"],
    "shots_total_per90": ["shots_total"],
    "passes_key_per90": ["passes_key"],
    "passes_total_per90": ["passes_total"],
    "dribbles_success_per90": ["dribbles_success"],
    "dribble_success_rate": ["dribbles_success", "dribbles_attempts"],
    "duels_won_per90": ["duels_won"],
    "duel_win_rate": ["duels_won", "duels_total"],
    "tackles_total_per90": ["tackles_total"],
    "interceptions_per90": ["interceptions"],
    "blocks_per90": ["blocks"],
    "fouls_drawn_per90": ["fouls_drawn"],
    "saves_per90": ["saves"],
    "share_shots": ["shots_total"],
    "share_dribbles": ["dribbles_attempts"],
    "share_passes": ["passes_total"],
    "share_duels": ["duels_total"],
}
COMPARED = [
    "minutes",
    "goals",
    "assists",
    "shots_total",
    "shots_on",
    "passes_total",
    "passes_key",
    "dribbles_attempts",
    "dribbles_success",
    "duels_total",
    "duels_won",
    "tackles_total",
    "interceptions",
    "blocks",
    "fouls_drawn",
    "fouls_committed",
    "dribbled_past",
    "saves",
    "cards_yellow",
]


def _corr(a: pd.Series, b: pd.Series) -> float:
    """Pearson correlation; missing when fewer than three pairs or no spread."""
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _sides(match: pd.DataFrame) -> pd.DataFrame:
    """One row per match: date, home and away team, score."""
    teams = match.drop_duplicates(["fixture_id", "team_id"])
    home = teams[teams["is_home"].astype(bool)]
    away = teams[~teams["is_home"].astype(bool)]
    out = home[["fixture_id", "kickoff", "team_id", "team_name", "team_goals"]].merge(
        away[["fixture_id", "team_id", "team_name", "team_goals"]],
        on="fixture_id",
        suffixes=("_home", "_away"),
    )
    out["day"] = out["kickoff"].dt.tz_convert("Europe/Madrid").dt.normalize().dt.tz_localize(None)
    return out


def link_matches(sb: pd.DataFrame, af: pd.DataFrame) -> pd.DataFrame:
    """sb_fixture and af_fixture of every StatsBomb match found in API
    Football: the same local day (one day either side), the same score, and
    the best mean similarity of the two club names (at least ``MIN_TEAM_SCORE``)."""
    s, a = _sides(sb), _sides(af)
    pairs = []
    for shift in (-1, 0, 1):
        moved = a.assign(day=a["day"] + pd.Timedelta(days=shift))
        pairs.append(s.merge(moved, on="day", suffixes=("_sb", "_af")))
    cand = pd.concat(pairs, ignore_index=True)
    cand = cand[
        (cand["team_goals_home_sb"] == cand["team_goals_home_af"])
        & (cand["team_goals_away_sb"] == cand["team_goals_away_af"])
    ].copy()

    def score(row) -> float:
        home = fuzz.token_set_ratio(
            normalize_club(row["team_name_home_sb"]), normalize_club(row["team_name_home_af"])
        )
        away = fuzz.token_set_ratio(
            normalize_club(row["team_name_away_sb"]), normalize_club(row["team_name_away_af"])
        )
        return (home + away) / 2

    cand["score"] = cand.apply(score, axis=1) if len(cand) else []
    cand = cand[cand["score"] >= MIN_TEAM_SCORE].sort_values("score", ascending=False)
    cand = cand.drop_duplicates("fixture_id_sb").drop_duplicates("fixture_id_af")
    return cand.rename(
        columns={
            "fixture_id_sb": "sb_fixture",
            "fixture_id_af": "af_fixture",
            "team_id_home_sb": "sb_home",
            "team_id_home_af": "af_home",
            "team_id_away_sb": "sb_away",
            "team_id_away_af": "af_away",
        }
    )[["sb_fixture", "af_fixture", "sb_home", "af_home", "sb_away", "af_away", "score"]]


def _name_ok(af_name: str, sb_name: str, sb_nick: str) -> float:
    af = normalize_name(af_name)
    best = max(
        fuzz.token_set_ratio(af, normalize_name(sb_name)),
        fuzz.token_set_ratio(af, normalize_name(sb_nick)),
    )
    surname = af.split()[-1] if af else ""
    if len(surname) >= 3 and surname in f"{normalize_name(sb_name)} {normalize_name(sb_nick)}":
        best = max(best, 100.0)
    return best


def link_players(sb: pd.DataFrame, af: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Rows of both sources for the same player and match: within a linked
    match and side, the same shirt number and a name similar enough; a
    player left over is linked by name alone (best score, at least 85)."""
    teams = pd.concat(
        [
            matches[["sb_fixture", "af_fixture", "sb_home", "af_home"]].set_axis(
                ["sb_fixture", "af_fixture", "sb_team", "af_team"], axis=1
            ),
            matches[["sb_fixture", "af_fixture", "sb_away", "af_away"]].set_axis(
                ["sb_fixture", "af_fixture", "sb_team", "af_team"], axis=1
            ),
        ]
    )
    s = sb.merge(teams, left_on=["fixture_id", "team_id"], right_on=["sb_fixture", "sb_team"])
    a = af.merge(teams, left_on=["fixture_id", "team_id"], right_on=["af_fixture", "af_team"])
    s = s.rename(columns={"source_player_id": "sb_player"})
    a = a.rename(columns={"source_player_id": "af_player"})
    joined = s[
        ["sb_fixture", "sb_team", "sb_player", "shirt_number", "player_name", "player_nickname"]
    ].merge(
        a[["sb_fixture", "af_player", "shirt_number", "player_name"]].rename(
            columns={"player_name": "af_name"}
        ),
        on=["sb_fixture", "shirt_number"],
    )
    joined = joined[
        joined.apply(
            lambda r: (
                _name_ok(r["af_name"], r["player_name"], r["player_nickname"] or "")
                >= MIN_NAME_SCORE
            ),
            axis=1,
        )
    ]
    linked = joined[["sb_fixture", "sb_player", "af_player"]].drop_duplicates()
    # Leftovers (a shirt number missing or different in one source): by name.
    rest_s = s[
        ~s.set_index(["sb_fixture", "sb_player"]).index.isin(
            linked.set_index(["sb_fixture", "sb_player"]).index
        )
    ]
    rest_a = a[
        ~a.set_index(["sb_fixture", "af_player"]).index.isin(
            linked.set_index(["sb_fixture", "af_player"]).index
        )
    ]
    extra = []
    by_match = {k: g for k, g in rest_a.groupby(["sb_fixture", "sb_team"])}
    for (fixture, team), group in rest_s.groupby(["sb_fixture", "sb_team"]):
        cands = by_match.get((fixture, team))
        if cands is None:
            continue
        for row in group.itertuples(index=False):
            scores = [
                (_name_ok(c.player_name, row.player_name, row.player_nickname or ""), c.af_player)
                for c in cands.itertuples(index=False)
            ]
            best = max(scores, default=(0.0, None))
            if best[0] >= 85:
                extra.append((fixture, row.sb_player, best[1]))
    if extra:
        linked = pd.concat(
            [linked, pd.DataFrame(extra, columns=["sb_fixture", "sb_player", "af_player"])]
        ).drop_duplicates(["sb_fixture", "sb_player"])
    return linked.merge(matches[["sb_fixture", "af_fixture"]], on="sb_fixture")


def player_ids(links: pd.DataFrame) -> pd.Series:
    """StatsBomb player id -> API Football player id, the most frequent link."""
    counts = links.groupby(["sb_player", "af_player"]).size().rename("n").reset_index()
    best = counts.sort_values(["n", "af_player"], ascending=[False, True]).drop_duplicates(
        "sb_player"
    )
    return best.set_index("sb_player")["af_player"]


def zeros_where_recorded(af: pd.DataFrame) -> pd.DataFrame:
    """API Football leaves a zero count blank: a blank becomes 0 for a player
    who played when anyone in that league season has the count."""
    out = af.copy()
    played = out["minutes"].fillna(0) > 0
    for c in COMPARED:
        recorded = out[c].notna().groupby([out["league_key"], out["season"]]).transform("any")
        out.loc[played & recorded & out[c].isna(), c] = 0
    return out


def paired_rows(sb: pd.DataFrame, af: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """The compared measures of both sources side by side (``<m>_sb``,
    ``<m>_af``) for linked players who played in both. API Football's blank
    counts are read as 0 where its league season records them for anyone,
    as the style inputs read them (``composite_inputs``)."""
    af = zeros_where_recorded(af)
    s = sb.rename(columns={"source_player_id": "sb_player", "fixture_id": "sb_fixture"})
    a = af.rename(columns={"source_player_id": "af_player", "fixture_id": "af_fixture"})
    s = s.assign(passes_complete=s["passes_accurate"])
    out = links.merge(
        s[["sb_fixture", "sb_player", "season", "passes_complete", *COMPARED]],
        on=["sb_fixture", "sb_player"],
    )
    out = out.merge(
        a[["af_fixture", "af_player", *COMPARED]],
        on=["af_fixture", "af_player"],
        suffixes=("_sb", "_af"),
    )
    played = (out["minutes_sb"].fillna(0) > 0) & (out["minutes_af"].fillna(0) > 0)
    return out[played].reset_index(drop=True)


def measure_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per measure: rows with both values, Pearson correlation, the two means
    and the mean difference (StatsBomb minus API Football)."""
    rows = []
    for m in COMPARED:
        a = pairs[f"{m}_sb"].astype(float)
        b = pairs[f"{m}_af"].astype(float)
        both = a.notna() & b.notna()
        r = _corr(a[both], b[both])
        rows.append(
            {
                "measure": m,
                "rows": int(both.sum()),
                "correlation": r,
                "mean_statsbomb": float(a[both].mean()),
                "mean_api_football": float(b[both].mean()),
                "mean_difference": float((a[both] - b[both]).mean()),
            }
        )
    return pd.DataFrame(rows)


def season_measures(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per season, the measures whose comparison moves between seasons:
    passes (also StatsBomb's completed passes against API Football's pass
    count), tackles, interceptions and blocks."""
    rows = []
    for season, g in pairs.groupby("season"):
        row = {"season": int(season), "rows": len(g)}
        for name, sb_col, af_col in (
            ("passes", "passes_total_sb", "passes_total_af"),
            ("completed_passes", "passes_complete", "passes_total_af"),
            ("tackles", "tackles_total_sb", "tackles_total_af"),
            ("interceptions", "interceptions_sb", "interceptions_af"),
            ("blocks", "blocks_sb", "blocks_af"),
        ):
            a, b = g[sb_col].astype(float), g[af_col].astype(float)
            both = a.notna() & b.notna()
            row[f"{name}_r"] = _corr(a[both], b[both])
            row[f"{name}_diff"] = float((a[both] - b[both]).mean()) if both.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def excluded_measures(table: pd.DataFrame) -> dict[str, list[str]]:
    """Style measure -> the counts behind it that correlate below
    ``MIN_CORRELATION`` (only measures with at least one)."""
    r = table.set_index("measure")["correlation"]
    out = {}
    for style, counts in STYLE_COUNTS.items():
        low = [c for c in counts if not (r.get(c, np.nan) >= MIN_CORRELATION)]
        if low:
            out[style] = low
    return out


def left_out(excluded: dict[str, list[str]]) -> dict[str, list[str]]:
    """The measures the StatsBomb vectors leave out: below the floor and not ``KEPT``."""
    return {m: c for m, c in excluded.items() if m not in KEPT}


def season_table(
    sb_rows: pd.DataFrame, af_rows: pd.DataFrame, ids: pd.Series, columns
) -> pd.DataFrame:
    """The style measures of the same player and season in both sources
    (450+ minutes in both): correlation per measure, a coarser check than
    the match one, on what the vectors read."""
    s = sb_rows[sb_rows["minutes"] >= 450].copy()
    s["af_player"] = s["source_player_id"].map(ids)
    s = s.dropna(subset=["af_player"])
    s["af_player"] = s["af_player"].astype(int)
    a = af_rows[(af_rows["league_key"] == "la_liga") & (af_rows["minutes"] >= 450)].copy()
    a["af_player"] = a["source_player_id"].astype(int)
    joined = s.merge(a, on=["af_player", "season"], suffixes=("_sb", "_af"))
    rows = []
    for c in columns:
        x, y = joined[f"{c}_sb"].astype(float), joined[f"{c}_af"].astype(float)
        both = x.notna() & y.notna()
        rows.append(
            {
                "measure": c,
                "player_seasons": int(both.sum()),
                "correlation": _corr(x[both], y[both]),
            }
        )
    return pd.DataFrame(rows)
