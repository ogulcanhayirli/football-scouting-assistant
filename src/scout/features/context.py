"""Context features from the tables beyond the league player stats: cup,
national team and youth matches, team strength, injuries, coaches and
transfers.

Point in time, like everything else: each feature only counts events dated
before the row's feature date, and each group keeps the date of the latest
event it used (``last_*_date``) so the contract can check that rule.

Where a source does not cover a row's window the feature is missing, not 0:
a player with no injury rows in a league season the API has no injury data
for is unknown, not fit.
"""

from __future__ import annotations

import unicodedata

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
    "youth_minutes_12m",
    "youth_matches_12m",
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
# The injuries table starts here; a 12 month window that begins earlier takes
# its injuries from the sidelined periods instead.
INJURIES_START = pd.Timestamp("2021-04-06")
# Sidelined periods: a snapshot of 2026 per player, dense from 2010 to
# September 2025; the API adds almost nothing after. A window must lie inside.
SIDELINED_START = pd.Timestamp("2010-07-01")
SIDELINED_END = pd.Timestamp("2025-10-01")
SPELL_GAP = pd.Timedelta(days=30)  # missed matches further apart start a new spell
# The sidelined endpoint tracks some league seasons only in part: 0 to 10% of
# the rows injured, where the injuries table finds about 45%. A league season
# takes its injuries from it only when at least this share has a spell.
SIDELINED_MIN_INJURED = 0.30
COACH_COLUMNS = ["new_coach", "coach_changed_in_season"]
OFF_SEASON_MONTHS = 2  # when the club's previous season is not held
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

    ``comps`` maps a competition key to (kind, region), kind "club",
    "national" or "youth"; ``league_regions`` maps a league key to its
    region. Club cup data starts at different times per region (UEFA 2015,
    CONMEBOL 2019; none held for North America and Asia), so a row's club cup
    columns are missing until its region's cups cover the whole window;
    national team and youth columns likewise, from the first match of their
    kind in the data. Inside a covered window no match means 0.

    Youth tournaments (U21 Euro, U20 and U17 World Cups, Olympics) get their
    own columns rather than counting as caps: a senior cap and an U21 match
    say different things about a player. The UEFA Youth League has lineups
    but no player minutes in the API, so it adds nothing here.
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
    youth = pairs[pairs["kind"] == "youth"].groupby("row")
    out["youth_minutes_12m"] = youth["minutes"].sum()
    out["youth_matches_12m"] = youth["minutes"].size()
    out["last_comp_match_date"] = pairs.groupby("row")["kickoff"].max()

    window_start = rows["feature_date"] - WINDOW
    club_start = m[m["kind"] == "club"].groupby("region")["kickoff"].min()
    # A dict: mapping through an empty datetime Series fails in pandas.
    row_club_start = rows["league_key"].map(league_regions).map(club_start.to_dict())
    row_club_start = pd.to_datetime(row_club_start).astype("datetime64[ns]")
    covered_by = {"club": window_start >= row_club_start}  # NaT start compares False
    for kind in ("national", "youth"):
        kickoffs = m.loc[m["kind"] == kind, "kickoff"]
        start = kickoffs.min() if len(kickoffs) else pd.NaT
        covered_by[kind] = window_start >= start if pd.notna(start) else False
    for kind, columns in (
        ("club", ["club_cup_minutes_12m", "club_cup_matches_12m", "ucl_minutes_12m"]),
        ("national", ["intl_minutes_12m", "intl_matches_12m", "intl_ga_12m"]),
        ("youth", ["youth_minutes_12m", "youth_matches_12m"]),
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


def club_calendar(elo_matches: pd.DataFrame) -> pd.DataFrame:
    """One row per club and match (league and club cup) from the Elo match
    table: team_id, fixture_id, kickoff (naive UTC), league_key, season, kind.
    The fixtures every held club played, from 2010, including seasons with no
    player stats."""
    columns = ["team_id", "fixture_id", "kickoff", "league_key", "season", "kind"]
    if elo_matches.empty:
        return pd.DataFrame(columns=columns)
    base = elo_matches[["fixture_id", "kickoff", "league_key", "season", "kind"]]
    sides = [base.assign(team_id=elo_matches[f"{side}_id"].to_numpy()) for side in ("home", "away")]
    out = pd.concat(sides, ignore_index=True)[columns]
    out["kickoff"] = _naive(pd.to_datetime(out["kickoff"], utc=True))
    out = out.astype({"team_id": "int64", "fixture_id": "int64", "season": "int64"})
    return out.drop_duplicates(["team_id", "fixture_id"]).reset_index(drop=True)


def _spells(pairs: pd.DataFrame, out: pd.DataFrame) -> None:
    """Fill the injury columns from (row, fixture_id, fixture_date) pairs of
    missed matches: a gap of more than ``SPELL_GAP`` starts a new spell."""
    pairs = pairs.drop_duplicates(["row", "fixture_id"]).sort_values(["row", "fixture_date"])
    gap = pairs.groupby("row")["fixture_date"].diff()
    pairs = pairs.assign(new_spell=gap.isna() | (gap > SPELL_GAP))
    grouped = pairs.groupby("row")
    out["injury_matches_missed_12m"] = grouped["fixture_id"].nunique()
    out["injury_spells_12m"] = grouped["new_spell"].sum()
    out["last_injury_date"] = grouped["fixture_date"].max()


def injury_coverage(injuries: pd.DataFrame) -> set[tuple[str, int]]:
    """League seasons the injuries table has rows for."""
    if injuries.empty:
        return set()
    pairs = injuries[["league_key", "season"]].drop_duplicates()
    return {(lg, int(se)) for lg, se in pairs.itertuples(index=False)}


def injury_features(
    rows: pd.DataFrame,
    injuries: pd.DataFrame,
    covered: set[tuple[str, int]],
    sidelined: pd.DataFrame | None = None,
    calendar: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Injury history in the 12 months before the feature date, from the
    injuries table where it covers the row, otherwise from the sidelined
    periods (``injury_source`` says which, missing when neither does).

    The injuries table covers a row when the row's own league season has
    injury data and the window starts on or after ``INJURIES_START``. The
    2020/21 windows start in mid 2020, nine months before the table, so they
    now take the sidelined periods too."""
    from_table = injury_table_features(rows, injuries, covered)
    table_rows = from_table["injury_spells_12m"].notna()
    table_rows &= rows["feature_date"] - WINDOW >= INJURIES_START
    out = from_table.where(table_rows)
    out["injury_source"] = pd.Series(pd.NA, index=rows.index, dtype="string")
    out.loc[table_rows, "injury_source"] = "injuries"
    if sidelined is None or sidelined.empty or calendar is None or calendar.empty:
        return out
    rest = rows[~table_rows]
    side = sidelined_injury_features(rest, sidelined, calendar)
    known = side["injury_spells_12m"].notna()
    out.loc[side.index[known], side.columns] = side[known]
    out.loc[side.index[known], "injury_source"] = "sidelined"
    out["last_injury_date"] = out["last_injury_date"].astype("datetime64[ns]")
    for column in INJURY_COLUMNS:
        out[column] = out[column].astype(float)
    return out


def injury_table_features(
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
    _spells(_window_events(rows, events, "fixture_date", WINDOW), out)
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


def sidelined_injury_features(
    rows: pd.DataFrame, sidelined: pd.DataFrame, calendar: pd.DataFrame
) -> pd.DataFrame:
    """The injury columns from the sidelined periods: the matches the row's
    club played (league and club cups) while the player was out, in the 12
    months before the feature date, and the spells they form.

    Point in time: only periods that started before the feature date count,
    and an end date on or after the feature date is never used: the period
    counts as still running at the feature date. Every player in the
    processed tables was asked for, so a player without periods is fit, but
    only for windows inside ``SIDELINED_START`` to ``SIDELINED_END`` and in
    league seasons the endpoint tracked (``SIDELINED_MIN_INJURED``); outside
    them, and for clubs without fixtures, the columns stay missing. The gate
    uses only periods started before the season's feature date."""
    out = pd.DataFrame(index=rows.index, columns=INJURY_COLUMNS, dtype=float)
    out["last_injury_date"] = pd.Series(pd.NaT, index=rows.index, dtype="datetime64[ns]")
    if rows.empty or sidelined.empty or calendar.empty:
        return out
    kind = sidelined["type"].fillna("").str.lower()
    periods = sidelined[~kind.str.contains(NOT_INJURY, regex=True)]
    periods = _plain(periods[["source_player_id", "start_date", "end_date"]], [])
    pairs = _window_events(rows, periods, "start_date", None)
    # Ends on or after the feature date are not known then: still running.
    running = pairs["end_date"].isna() | (pairs["end_date"] >= pairs["feature_date"])
    pairs["until"] = pairs["end_date"].where(~running, pairs["feature_date"])
    pairs["since"] = pairs["start_date"].clip(lower=pairs["feature_date"] - WINDOW)
    pairs = pairs[pairs["until"] > pairs["since"]]
    team = rows["team_id"].reindex(pairs["row"]).to_numpy()
    pairs = pairs.assign(team_id=pd.to_numeric(team, errors="coerce")).dropna(subset=["team_id"])
    pairs["team_id"] = pairs["team_id"].astype("int64")
    games = pairs.merge(calendar[["team_id", "fixture_id", "kickoff"]], on="team_id")
    inside = (games["kickoff"] >= games["since"]) & (games["kickoff"] < games["until"])
    games = games[inside].rename(columns={"kickoff": "fixture_date"})
    _spells(games[["row", "fixture_id", "fixture_date"]], out)

    fd = rows["feature_date"]
    in_range = (fd - WINDOW >= SIDELINED_START) & (fd <= SIDELINED_END)
    team_ids = pd.to_numeric(rows["team_id"], errors="coerce")
    has_club = team_ids.isin(set(calendar["team_id"]))
    known = in_range & has_club
    hurt = (out["injury_spells_12m"].fillna(0) > 0).where(known)
    share = hurt.groupby([rows["league_key"], rows["season"]]).transform("mean")
    known &= share >= SIDELINED_MIN_INJURED
    for column in INJURY_COLUMNS:
        out[column] = out[column].fillna(0).where(known)
    out.loc[~known, "last_injury_date"] = pd.NaT
    return out


def injury_source_agreement(
    rows: pd.DataFrame,
    injuries: pd.DataFrame,
    covered: set[tuple[str, int]],
    sidelined: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Per season, the rows both sources cover, the share injured (one spell
    or more) by each, how often they agree on that, and the rank correlation
    of matches missed."""
    table = injury_table_features(rows, injuries, covered)
    side = sidelined_injury_features(rows, sidelined, calendar)
    both = table["injury_spells_12m"].notna() & side["injury_spells_12m"].notna()
    frame = pd.DataFrame(
        {
            "season": rows["season"].astype(int),
            "table_hurt": table["injury_spells_12m"] > 0,
            "side_hurt": side["injury_spells_12m"] > 0,
            "table_missed": table["injury_matches_missed_12m"],
            "side_missed": side["injury_matches_missed_12m"],
        }
    )[both]
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby("season")
    out = grouped.agg(
        rows=("table_hurt", "size"),
        injured_injuries=("table_hurt", "mean"),
        injured_sidelined=("side_hurt", "mean"),
        missed_injuries=("table_missed", "mean"),
        missed_sidelined=("side_missed", "mean"),
    )
    out["agree"] = grouped.apply(lambda g: (g["table_hurt"] == g["side_hurt"]).mean())
    out["spearman_missed"] = grouped.apply(
        lambda g: g["table_missed"].corr(g["side_missed"], method="spearman")
    )
    return out.reset_index()


# Coaches


def coach_features(
    rows: pd.DataFrame, coaches: pd.DataFrame, calendar: pd.DataFrame
) -> pd.DataFrame:
    """Per row, for the club and league season of the row:

    * ``new_coach``: a coach spell of the club started after the month of the
      club's last league match of the previous season and no later than the
      month of its first league match of this one (the coach at the start is
      not the one who finished last season). When the club's previous season
      is not held, the off season is taken as the ``OFF_SEASON_MONTHS``
      months before the first match.
    * ``coach_changed_in_season``: a spell started after the month of the
      first league match and no later than the month of the last one.

    Spell dates have month precision, so both flags compare months; a coach
    appointed in the month of the first match counts as new for the season,
    not as an in season change. Caretakers count; a spell that only
    re-lists the coach already in charge does not (``_coach_changes``).
    Only start dates are read for the flags,
    and every start used falls on or before the month of the last league
    match and before the feature date; ``last_coach_change_date`` keeps the
    latest one for the contract. Missing for clubs with no spell starting by
    then, or without league fixtures that season."""
    out = pd.DataFrame(index=rows.index, columns=COACH_COLUMNS, dtype=float)
    out["last_coach_change_date"] = pd.Series(pd.NaT, index=rows.index, dtype="datetime64[ns]")
    if rows.empty or coaches.empty or calendar.empty:
        return out
    league = calendar[calendar["kind"] == "league"]
    # The club's last league match of each season, in whichever league.
    prev = league.groupby(["team_id", "season"])["kickoff"].max().dt.to_period("M")
    prev = prev.rename("prev_last_m").reset_index()
    prev["season"] += 1

    keys = rows[["league_key", "season", "team_id", "feature_date"]].reset_index(names="row")
    keys = keys.dropna(subset=["team_id"]).astype({"season": "int64", "team_id": "int64"})
    # Only fixtures before the feature date say when the season ran.
    played = league.merge(
        keys[["league_key", "season", "team_id", "feature_date"]].drop_duplicates(),
        on=["league_key", "season", "team_id"],
    )
    played = played[played["kickoff"] < played["feature_date"]]
    span = (
        played.groupby(["league_key", "season", "team_id"])["kickoff"]
        .agg(first="min", last="max")
        .reset_index()
    )
    span["first_m"] = span["first"].dt.to_period("M")
    span["last_m"] = span["last"].dt.to_period("M")
    keys = keys.merge(span, on=["league_key", "season", "team_id"]).merge(
        prev, on=["team_id", "season"], how="left"
    )
    fallback = keys["first_m"] - OFF_SEASON_MONTHS
    keys["prev_last_m"] = keys["prev_last_m"].where(
        keys["prev_last_m"].notna() & (keys["prev_last_m"] < keys["first_m"]), fallback
    )

    spells = _coach_changes(coaches)
    spells["start_m"] = spells["start_date"].dt.to_period("M")
    pairs = keys.merge(spells, on="team_id")
    # Month level for the flags, but a start dated after the feature date is
    # never known (a few spells carry a day: Rosario Central, 20 May 2018,
    # after the 16 May feature date of Liga Profesional 2017).
    pairs = pairs[
        (pairs["start_m"] <= pairs["last_m"]) & (pairs["start_date"] < pairs["feature_date"])
    ]
    new = (pairs["start_m"] > pairs["prev_last_m"]) & (pairs["start_m"] <= pairs["first_m"])
    during = (pairs["start_m"] > pairs["first_m"]) & (pairs["start_m"] <= pairs["last_m"])
    new &= pairs["change"]
    during &= pairs["change"]
    pairs = pairs.assign(new=new, during=during)
    grouped = pairs.groupby("row")
    out["new_coach"] = grouped["new"].any().astype(float)
    out["coach_changed_in_season"] = grouped["during"].any().astype(float)
    used = pairs[new | during]
    out["last_coach_change_date"] = used.groupby("row")["start_date"].max()
    return out


def _person(name: object, coach_id: object) -> str:
    """First initial and surname, without accents: "J. Neestrup" and "Jacob
    Neestrup" are one person. The id stands in when the name is missing."""
    if not isinstance(name, str) or not name.strip():
        return f"id:{coach_id}"
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    parts = plain.replace(".", " ").split()
    return f"{parts[0][0]} {parts[-1]}" if parts else f"id:{coach_id}"


def _coach_changes(coaches: pd.DataFrame) -> pd.DataFrame:
    """team_id, start_date and ``change`` per spell. From July 2025 the API
    lists coaches already in charge again under a new id and their full name
    (FC Copenhagen: J. Neestrup since September 2022, Jacob Neestrup from
    July 2025), which would read as an appointment at almost every club. A
    spell is no change when the same person already has a spell at the club
    that started earlier and had not ended a month before this one began."""
    spells = coaches.dropna(subset=["team_id", "start_date"]).copy()
    spells["team_id"] = spells["team_id"].astype("int64")
    spells["person"] = [
        _person(n, c) for n, c in zip(spells["coach_name"], spells["coach_id"], strict=True)
    ]
    spells = spells.reset_index(drop=True)
    earlier = spells.reset_index(names="spell").merge(
        spells[["team_id", "person", "start_date", "end_date"]],
        on=["team_id", "person"],
        suffixes=("", "_before"),
    )
    running = (earlier["start_date_before"] < earlier["start_date"]) & (
        earlier["end_date_before"].isna()
        | (earlier["end_date_before"] >= earlier["start_date"] - pd.DateOffset(months=1))
    )
    continued = set(earlier.loc[running, "spell"])
    spells["change"] = ~spells.index.isin(continued)
    return spells[["team_id", "start_date", "change"]].drop_duplicates()


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
    events = _plain(
        transfers[["source_player_id", "transfer_date", "kind", "fee_eur"]], ["fee_eur"]
    )
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
