"""Context features from the tables beyond the league player stats: cup and
national team matches, team strength, injuries and transfers.

Point in time, like everything else: each feature only counts events dated
before the row's feature date, and each group keeps the date of the latest
event it used (``last_*_date``) so the contract can check that rule.

Where a source does not cover a row's window the feature is missing, not 0:
a player with no injury rows in a league season the API has no injury data
for is unknown, not fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = pd.Timedelta(days=365)
# Injury reasons that are not injuries: suspensions and absences.
NOT_INJURY = r"card|suspen|inactive|personal|national|coach|rest"

# Columns added by each group, for the docs and the column order.
COMPETITION_COLUMNS = [
    "club_cup_minutes_12m",
    "club_cup_matches_12m",
    "club_cup_ga_per90_12m",
    "club_cup_rating_12m",
    "ucl_minutes_12m",
    "intl_minutes_12m",
    "intl_matches_12m",
    "intl_ga_12m",
]
TEAM_COLUMNS = [
    "team_possession",
    "team_shots_pm",
    "team_shots_against_pm",
    "team_shot_share",
    "team_pass_accuracy",
    "team_xg_pm",
    "team_xga_pm",
]
INJURY_COLUMNS = ["injury_matches_missed_12m", "injury_spells_12m"]
TRANSFER_COLUMNS = [
    "career_moves",
    "career_loans",
    "career_paid_moves",
    "moves_last_2y",
    "days_since_last_move",
    "last_move_kind",
    "last_fee_eur",
    "max_fee_eur_to_date",
    "on_loan",
]


def _window_events(
    rows: pd.DataFrame, events: pd.DataFrame, date_column: str, window: pd.Timedelta | None
) -> pd.DataFrame:
    """Every (row, event) pair of the same player with the event before the
    row's feature date and, when ``window`` is given, inside it. Keeps the
    row's index as ``row``."""
    left = rows[["source_player_id", "feature_date"]].reset_index(names="row")
    left = left.dropna(subset=["source_player_id"])
    left["source_player_id"] = left["source_player_id"].astype("int64")
    left["feature_date"] = left["feature_date"].astype("datetime64[ns]")
    pairs = left.merge(events, on="source_player_id")
    before = pairs[date_column] < pairs["feature_date"]
    if window is not None:
        before &= pairs[date_column] >= pairs["feature_date"] - window
    return pairs[before]


def _plain(frame: pd.DataFrame, numbers: list[str]) -> pd.DataFrame:
    """Plain numpy dtypes: a player id as int64, the rest as float. Nullable
    pandas dtypes fail to align in the window joins below."""
    frame = frame.dropna(subset=["source_player_id"]).copy()
    frame["source_player_id"] = frame["source_player_id"].astype("int64")
    for column in numbers:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    return frame


def _naive(series: pd.Series) -> pd.Series:
    if getattr(series.dt, "tz", None) is not None:
        series = series.dt.tz_convert(None)
    return series.astype("datetime64[ns]")


# Cups and national teams


def competition_features(
    rows: pd.DataFrame,
    comp_matches: pd.DataFrame,
    comps: dict[str, tuple[str, str]],
    league_regions: dict[str, str],
) -> pd.DataFrame:
    """Minutes, matches and output in club cups and for the national team in
    the 12 months before the feature date.

    ``comps`` maps a competition key to (kind, region), kind "club" or
    "national"; ``league_regions`` maps a league key to its region. Club cup
    data starts at different times per region (UEFA 2015, CONMEBOL 2019), so
    a row's club cup columns are missing until its region's cups cover the
    whole window; national team columns likewise, from the first national
    team match in the data. Inside a covered window no match means 0.
    """
    out = pd.DataFrame(index=rows.index, columns=COMPETITION_COLUMNS, dtype=float)
    out["last_comp_match_date"] = pd.Series(pd.NaT, index=rows.index, dtype="datetime64[ns]")
    if comp_matches.empty:
        return out
    m = _plain(comp_matches, ["minutes", "goals", "assists", "rating"])
    m = m[m["minutes"].fillna(0) > 0]
    m["kickoff"] = _naive(m["kickoff"])
    m["kind"] = m["league_key"].map(lambda k: comps[k][0])
    m["region"] = m["league_key"].map(lambda k: comps[k][1])
    m["ga"] = m["goals"].fillna(0) + m["assists"].fillna(0)
    m["rating_x_min"] = m["rating"] * m["minutes"]
    m["rated_min"] = m["minutes"].where(m["rating"].notna())
    events = m[
        [
            "source_player_id",
            "kickoff",
            "kind",
            "league_key",
            "minutes",
            "ga",
            "rating_x_min",
            "rated_min",
        ]
    ]
    pairs = _window_events(rows, events, "kickoff", WINDOW)

    club = pairs[pairs["kind"] == "club"].groupby("row")
    out["club_cup_minutes_12m"] = club["minutes"].sum()
    out["club_cup_matches_12m"] = club["minutes"].size()
    out["club_cup_ga_per90_12m"] = club["ga"].sum() / out["club_cup_minutes_12m"] * 90
    out["club_cup_rating_12m"] = club["rating_x_min"].sum() / club["rated_min"].sum().replace(
        0, np.nan
    )
    ucl = pairs[pairs["league_key"] == "champions_league"].groupby("row")
    out["ucl_minutes_12m"] = ucl["minutes"].sum()
    intl = pairs[pairs["kind"] == "national"].groupby("row")
    out["intl_minutes_12m"] = intl["minutes"].sum()
    out["intl_matches_12m"] = intl["minutes"].size()
    out["intl_ga_12m"] = intl["ga"].sum()
    out["last_comp_match_date"] = pairs.groupby("row")["kickoff"].max()

    window_start = rows["feature_date"] - WINDOW
    club_start = m[m["kind"] == "club"].groupby("region")["kickoff"].min()
    row_club_start = rows["league_key"].map(league_regions).map(club_start)
    national = m.loc[m["kind"] == "national", "kickoff"]
    national_start = national.min() if len(national) else pd.NaT
    covered_by = {
        "club": window_start >= row_club_start,  # NaT start compares False
        "national": window_start >= national_start if pd.notna(national_start) else False,
    }
    for kind, columns in (
        ("club", ["club_cup_minutes_12m", "club_cup_matches_12m", "ucl_minutes_12m"]),
        ("national", ["intl_minutes_12m", "intl_matches_12m", "intl_ga_12m"]),
    ):
        covered = pd.Series(covered_by[kind], index=rows.index).fillna(False).astype(bool)
        for column in columns:
            out.loc[covered, column] = out.loc[covered, column].fillna(0)
            out.loc[~covered, column] = np.nan
    return out


# Team strength


def team_strength(team_stats: pd.DataFrame, dates: pd.DataFrame) -> pd.DataFrame:
    """Per league, season and team, over league matches before the feature
    date: possession, shots for and against per match, share of the shots,
    pass accuracy, and xG for and against (recorded from 2023/24 only)."""
    keys = ["league_key", "season", "team_id"]
    if team_stats.empty:
        empty = pd.DataFrame({k: pd.Series(dtype="Int64") for k in keys})
        empty["league_key"] = empty["league_key"].astype("string")
        return empty.assign(**{c: pd.Series(dtype=float) for c in TEAM_COLUMNS})
    t = team_stats.merge(dates, on=["league_key", "season"])
    t = t[_naive(t["kickoff"]) < t["feature_date"]]
    opponent = t[["fixture_id", "team_id", "shots_total", "xg"]].rename(
        columns={"team_id": "opponent_id", "shots_total": "shots_against", "xg": "xga"}
    )
    t = t.merge(opponent, on=["fixture_id", "opponent_id"], how="left")
    grouped = t.groupby(keys)
    out = grouped.agg(
        team_possession=("possession", "mean"),
        team_shots_pm=("shots_total", "mean"),
        team_shots_against_pm=("shots_against", "mean"),
        team_pass_accuracy=("pass_accuracy", "mean"),
        team_xg_pm=("xg", "mean"),
        team_xga_pm=("xga", "mean"),
    )
    total = out["team_shots_pm"] + out["team_shots_against_pm"]
    out["team_shot_share"] = out["team_shots_pm"] / total.where(total > 0)
    for column in TEAM_COLUMNS:
        out[column] = out[column].astype(float)
    return out[TEAM_COLUMNS].reset_index()


# Injuries


def injury_features(
    rows: pd.DataFrame, injuries: pd.DataFrame, covered: set[tuple[str, int]]
) -> pd.DataFrame:
    """Matches missed through injury in the 12 months before the feature
    date, across every league and cup, and the number of separate spells (a
    gap of more than 30 days between missed matches starts a new one).
    Suspensions are not injuries. Missing where the row's own league season
    has no injury data."""
    out = pd.DataFrame(index=rows.index, columns=INJURY_COLUMNS, dtype=float)
    out["last_injury_date"] = pd.Series(pd.NaT, index=rows.index, dtype="datetime64[ns]")
    if injuries.empty:
        return out
    i = injuries[injuries["status"] == "Missing Fixture"]
    reason = i["reason"].fillna("").str.lower()
    i = i[~reason.str.contains(NOT_INJURY, regex=True)]
    events = _plain(i[["source_player_id", "fixture_date", "fixture_id"]], ["fixture_id"])
    events["fixture_date"] = _naive(events["fixture_date"])
    events = events.drop_duplicates(["source_player_id", "fixture_id"])
    pairs = _window_events(rows, events, "fixture_date", WINDOW).sort_values(
        ["row", "fixture_date"]
    )
    gap = pairs.groupby("row")["fixture_date"].diff()
    pairs["new_spell"] = gap.isna() | (gap > pd.Timedelta(days=30))
    grouped = pairs.groupby("row")
    out["injury_matches_missed_12m"] = grouped["fixture_id"].nunique()
    out["injury_spells_12m"] = grouped["new_spell"].sum()
    out["last_injury_date"] = grouped["fixture_date"].max()
    has_data = pd.Series(
        [
            (lg, int(se)) in covered
            for lg, se in zip(rows["league_key"], rows["season"], strict=True)
        ],
        index=rows.index,
    )
    for column in INJURY_COLUMNS:
        out[column] = out[column].fillna(0).where(has_data)
    return out


# Transfers


def transfer_features(rows: pd.DataFrame, transfers: pd.DataFrame) -> pd.DataFrame:
    """The player's moves before the feature date: how many, how many loans
    and paid moves, how recent, the last one's kind and fee, the highest fee
    so far, and whether the last move was a loan out. Missing for players
    with no transfer history at all (the API lists none for them)."""
    out = pd.DataFrame(index=rows.index, columns=TRANSFER_COLUMNS, dtype=float)
    out["last_move_kind"] = pd.Series(pd.NA, index=rows.index, dtype="string")
    out["last_transfer_date"] = pd.Series(pd.NaT, index=rows.index, dtype="datetime64[ns]")
    if transfers.empty:
        return out
    events = _plain(transfers[["source_player_id", "transfer_date", "kind", "fee_eur"]], ["fee_eur"])
    events["transfer_date"] = _naive(events["transfer_date"])
    pairs = _window_events(rows, events, "transfer_date", None).sort_values(
        ["row", "transfer_date"]
    )
    grouped = pairs.groupby("row")
    out["career_moves"] = grouped.size()
    out["career_loans"] = grouped["kind"].agg(lambda k: (k == "loan").sum())
    out["career_paid_moves"] = grouped["fee_eur"].agg(lambda f: (f > 0).sum())
    recent = pairs[pairs["transfer_date"] >= pairs["feature_date"] - 2 * WINDOW]
    out["moves_last_2y"] = recent.groupby("row").size()
    last = grouped.tail(1).set_index("row")
    out["last_transfer_date"] = last["transfer_date"]
    out["days_since_last_move"] = (last["feature_date"] - last["transfer_date"]).dt.days
    out["last_move_kind"] = last["kind"].astype("string")
    out["last_fee_eur"] = last["fee_eur"].astype(float)
    out["max_fee_eur_to_date"] = grouped["fee_eur"].max().astype(float)
    out["on_loan"] = (last["kind"] == "loan").astype(float)  # 1 or 0; a float for the trees

    known = rows["source_player_id"].isin(set(transfers["source_player_id"]))
    for column in ("career_moves", "career_loans", "career_paid_moves", "moves_last_2y"):
        out[column] = out[column].fillna(0).where(known)
    out.loc[known & out["on_loan"].isna(), "on_loan"] = 0.0
    return out
