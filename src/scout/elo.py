"""Team Elo from our own fixture results, point in time.

ClubElo is not used: its API went offline in September 2026 (decision 7).

* **Matches:** every finished club match in the raw ``/fixtures`` cache, from
  2010: each domestic league we hold (the configured leagues and the second
  tiers and feeder leagues of the expansion probe) and the five club cups,
  which link the leagues (Champions League, Europa League, Conference League,
  Libertadores, Sudamericana). National team and youth matches are left out:
  those teams never meet a club, so they would only blur the home advantage.
  The result is the score after 90 minutes; extra time and penalties are not
  counted. Awarded matches (``AWD``) are not football and are skipped.
* **Update:** expected score ``1 / (1 + 10^(-(R_home + H - R_away) / 400))``;
  each side moves by ``K x G x (result - expected)``, where ``G`` weights the
  goal difference as eloratings.net does: 1 for a margin of 0 or 1, 1.5 for
  2, ``(11 + margin) / 8`` above.
* **League link:** a cup match between clubs of two different leagues is
  evidence about both leagues, not only about the two clubs. Every other club
  of each league moves by ``spread`` times its club's change. Without it,
  a few dozen cup matches a season would take decades to separate leagues
  that all started at 1500.
* **New clubs** start at the lower quartile of their league at that moment
  (most are promoted from a tier we do not hold), or at the lower quartile
  of all clubs when their league is not held (cup entrants).
* **Fit:** home advantage ``H``, ``K`` and ``spread`` by grid search on the
  Brier score of the expected result, scored only on matches before
  ``FIT_END``, the start of the first season with match tier feature rows.
  The settings therefore use no result that a feature row could see later.
* **Point in time:** every match stores both clubs' ratings before kickoff.
  A club's season rating is the mean of its pre match ratings over its league
  matches that season; all of them precede the season's feature date.
"""

from __future__ import annotations

import csv
import itertools
import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CLUB_CUPS = {2: "champions_league", 3: "europa_league", 848: "conference_league"}
CLUB_CUPS |= {13: "copa_libertadores", 11: "copa_sudamericana"}
COUNTED = {"FT", "AET", "PEN"}  # finished on the pitch; AWD is a desk decision
START = 1500.0
BURN_IN_END = pd.Timestamp("2012-07-01", tz="UTC")  # matches before this only warm up
FIT_END = pd.Timestamp("2015-07-01", tz="UTC")  # first match tier feature season is 2015
GRID = {
    "home_advantage": [50.0, 60.0, 70.0, 80.0, 90.0],
    "k": [10.0, 12.0, 15.0, 20.0],
    "spread": [0.0, 0.5, 0.75, 1.0, 1.5],
}


@dataclass(frozen=True)
class EloParams:
    home_advantage: float = 70.0
    k: float = 12.0
    spread: float = 1.0


# Loading


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def league_keys(settings, probe_csv: Path | None = None) -> dict[int, str]:
    """API league id to our key: the config, then the expansion probe."""
    keys = {lg.api_football_id: lg.key for lg in settings.leagues}
    keys |= {c.api_football_id: c.key for c in settings.competitions}
    if probe_csv is not None and probe_csv.is_file():
        with probe_csv.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                keys.setdefault(int(row["league_id"]), row["key"])
    return keys


def parse_fixtures(payload: dict, keys: dict[int, str] | None = None) -> list[dict]:
    """Club matches from one ``/fixtures`` response, one row per match.

    A league counts as domestic when the API gives it a country; ``World``
    competitions count only if they are one of the club cups.
    """
    keys = keys or {}
    rows = []
    for f in payload.get("response") or []:
        fixture, league = f.get("fixture") or {}, f.get("league") or {}
        if ((fixture.get("status") or {}).get("short")) not in COUNTED:
            continue
        league_id = league.get("id")
        if league_id in CLUB_CUPS:
            kind = "cup"
        elif league.get("country") and league.get("country") != "World":
            kind = "league"
        else:
            continue
        full = (f.get("score") or {}).get("fulltime") or {}
        goals = f.get("goals") or {}
        home_goals = full.get("home", goals.get("home"))
        away_goals = full.get("away", goals.get("away"))
        teams = f.get("teams") or {}
        home, away = teams.get("home") or {}, teams.get("away") or {}
        if home_goals is None or away_goals is None or not home.get("id") or not away.get("id"):
            log.warning("fixture %s: finished without a score or teams; skipped", fixture.get("id"))
            continue
        rows.append(
            {
                "fixture_id": int(fixture["id"]),
                "kickoff": fixture.get("date"),
                "league_id": int(league_id),
                "league_key": keys.get(
                    league_id, _slug(f"{league.get('country', '')} {league.get('name', '')}")
                ),
                "season": int(league.get("season")),
                "kind": kind,
                "home_id": int(home["id"]),
                "home_name": home.get("name"),
                "away_id": int(away["id"]),
                "away_name": away.get("name"),
                "home_goals": int(home_goals),
                "away_goals": int(away_goals),
            }
        )
    return rows


def load_matches(raw_dir: Path, keys: dict[int, str] | None = None) -> pd.DataFrame:
    """Every cached ``/fixtures`` league season response, as club matches.
    Reads the cache only: no request is made."""
    rows: list[dict] = []
    base = Path(raw_dir) / "api_football" / "fixtures"
    for path in sorted(base.glob("league=*/season=*.json")):
        if path.name.endswith(".meta.json"):
            continue
        rows += parse_fixtures(json.loads(path.read_bytes()), keys)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["kickoff"] = pd.to_datetime(frame["kickoff"], utc=True)
    frame = frame.drop_duplicates("fixture_id")
    return frame.sort_values(["kickoff", "fixture_id"]).reset_index(drop=True)


# Rating


def goal_weight(margin: np.ndarray | int) -> np.ndarray:
    m = np.abs(np.asarray(margin, dtype=float))
    return np.where(m <= 1, 1.0, np.where(m == 2, 1.5, (11.0 + m) / 8.0))


def expected(diff: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + 10.0 ** (-np.asarray(diff, dtype=float) / 400.0))


def run_elo(matches: pd.DataFrame, params: EloParams) -> pd.DataFrame:
    """Pre match ratings for every match, in kickoff order.

    Adds ``home_pre``, ``away_pre`` (ratings before kickoff), ``expected_home``,
    ``home_post``, ``away_post``, the clubs' domestic leagues at kickoff
    (``home_league``, ``away_league``) and ``cross_league``.
    """
    m = matches.sort_values(["kickoff", "fixture_id"]).reset_index(drop=True)
    teams = pd.Index(pd.unique(np.concatenate([m["home_id"], m["away_id"]])))
    hi, ai = teams.get_indexer(m["home_id"]), teams.get_indexer(m["away_id"])
    league_codes, league_names = pd.factorize(m["league_key"])
    is_league = (m["kind"] == "league").to_numpy()
    hg, ag = m["home_goals"].to_numpy(), m["away_goals"].to_numpy()
    result = np.where(hg > ag, 1.0, np.where(hg == ag, 0.5, 0.0))
    weight = goal_weight(hg - ag)

    n = len(teams)
    rating = np.full(n, np.nan)
    home_league = np.full(n, -1)  # each club's latest domestic league, as of now
    out = np.empty((len(m), 7))
    for i in range(len(m)):
        h, a = hi[i], ai[i]
        if is_league[i]:
            home_league[h] = home_league[a] = league_codes[i]
        for t in (h, a):
            if np.isnan(rating[t]):
                rating[t] = _start(rating, home_league, home_league[t])
        rh, ra = rating[h], rating[a]
        e = float(expected(rh + params.home_advantage - ra))
        delta = params.k * weight[i] * (result[i] - e)
        lh, la = home_league[h], home_league[a]
        cross = (not is_league[i]) and lh != la
        if cross and params.spread:
            if lh >= 0:
                rating[home_league == lh] += params.spread * delta
            if la >= 0:
                rating[home_league == la] -= params.spread * delta
            # the two clubs themselves get the full change, not full plus spread
            rating[h] -= params.spread * delta if lh >= 0 else 0.0
            rating[a] += params.spread * delta if la >= 0 else 0.0
        rating[h] += delta
        rating[a] -= delta
        out[i] = (rh, ra, e, rating[h], rating[a], lh, la)
    m["home_pre"], m["away_pre"], m["expected_home"] = out[:, 0], out[:, 1], out[:, 2]
    m["home_post"], m["away_post"] = out[:, 3], out[:, 4]
    names = np.append(np.asarray(league_names, dtype=object), None)
    m["home_league"] = names[out[:, 5].astype(int)]
    m["away_league"] = names[out[:, 6].astype(int)]
    m["cross_league"] = (~is_league) & (out[:, 5] != out[:, 6])
    m["result_home"] = result
    return m


def _start(rating: np.ndarray, league_of: np.ndarray, league: int) -> float:
    rated = ~np.isnan(rating)
    peers = rating[rated & (league_of == league)] if league >= 0 else np.array([])
    if len(peers) >= 5:
        return float(np.quantile(peers, 0.25))
    if league < 0 and rated.sum() >= 5:
        return float(np.quantile(rating[rated], 0.25))
    return START


def brier(rated: pd.DataFrame, start=BURN_IN_END, end=FIT_END) -> dict[str, float]:
    """Brier score of the expected result, all matches and cross league ones."""
    window = rated[(rated["kickoff"] >= start) & (rated["kickoff"] < end)]
    err = (window["result_home"] - window["expected_home"]) ** 2
    cross = window["cross_league"].to_numpy()
    return {
        "matches": len(window),
        "brier_all": float(err.mean()),
        "brier_domestic": float(err[~cross].mean()),
        "cross_matches": int(cross.sum()),
        "brier_cross": float(err[cross].mean()) if cross.any() else float("nan"),
    }


def fit(
    matches: pd.DataFrame, grid: dict[str, Iterable[float]] = GRID
) -> tuple[EloParams, pd.DataFrame]:
    """Grid search on matches before ``FIT_END``. The objective weighs the
    domestic and the cross league Brier score equally: the cross league
    matches are few but carry the whole link between leagues."""
    early = matches[matches["kickoff"] < FIT_END]
    rows = []
    for h, k, s in itertools.product(grid["home_advantage"], grid["k"], grid["spread"]):
        p = EloParams(h, k, s)
        score = brier(run_elo(early, p))
        rows.append({"home_advantage": h, "k": k, "spread": s, **score})
    table = pd.DataFrame(rows)
    table["objective"] = table[["brier_domestic", "brier_cross"]].mean(axis=1)  # skips NaN
    table = table.sort_values("objective").reset_index(drop=True)
    best = table.iloc[0]
    params = EloParams(*(float(best[c]) for c in ("home_advantage", "k", "spread")))
    return params, table


# Tables


def match_long(rated: pd.DataFrame) -> pd.DataFrame:
    """One row per club per match: pre and post match rating."""
    home = rated.rename(
        columns={
            "home_id": "team_id",
            "home_name": "team_name",
            "home_pre": "elo_pre",
            "home_post": "elo_post",
            "home_league": "team_league",
        }
    )
    away = rated.rename(
        columns={
            "away_id": "team_id",
            "away_name": "team_name",
            "away_pre": "elo_pre",
            "away_post": "elo_post",
            "away_league": "team_league",
        }
    )
    keep = [
        "fixture_id",
        "kickoff",
        "league_key",
        "season",
        "kind",
        "team_id",
        "team_name",
        "team_league",
        "elo_pre",
        "elo_post",
    ]
    long = pd.concat([home[keep], away[keep]], ignore_index=True)
    return long.sort_values(["kickoff", "fixture_id"]).reset_index(drop=True)


def rating_asof(long: pd.DataFrame, team_id: int, when) -> float:
    """A club's rating on a date: after its last match strictly before it."""
    when = pd.Timestamp(when)
    when = when.tz_localize("UTC") if when.tzinfo is None else when
    rows = long[(long["team_id"] == team_id) & (long["kickoff"] < when)]
    return float(rows["elo_post"].iloc[-1]) if len(rows) else float("nan")


def team_seasons(long: pd.DataFrame) -> pd.DataFrame:
    """Per club and league season: mean pre match rating over its league
    matches, the rating after the last one, and that match's date."""
    league = long[long["kind"] == "league"]
    g = league.groupby(["league_key", "season", "team_id"])
    out = g.agg(
        team_name=("team_name", "last"),
        matches=("elo_pre", "size"),
        elo=("elo_pre", "mean"),
        elo_end=("elo_post", "last"),
        last_match=("kickoff", "max"),
    ).reset_index()
    return out


def league_seasons(teams: pd.DataFrame) -> pd.DataFrame:
    """League strength: the mean season rating of the league's clubs."""
    g = teams.groupby(["league_key", "season"])
    return g.agg(
        clubs=("team_id", "size"), elo=("elo", "mean"), elo_end=("elo_end", "mean")
    ).reset_index()


def build_elo(settings, params: EloParams | None = None):
    """Load, fit (unless ``params`` is given) and rate. Returns
    ``(params, fit_table, rated_matches, team_seasons, league_seasons)``."""
    keys = league_keys(settings, Path("configs/expansion_probe.csv"))
    matches = load_matches(settings.paths.raw_dir, keys)
    log.info(
        "elo: %d club matches, %d clubs",
        len(matches),
        pd.concat([matches["home_id"], matches["away_id"]]).nunique(),
    )
    fit_table = None
    if params is None:
        params, fit_table = fit(matches)
        log.info("elo: fitted %s", params)
    rated = run_elo(matches, params)
    teams = team_seasons(match_long(rated))
    return params, fit_table, rated, teams, league_seasons(teams)


def write_tables(out_dir: Path, rated, teams, leagues) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rated.to_parquet(out_dir / "elo_matches.parquet", index=False)
    teams.to_parquet(out_dir / "team_season_elo.parquet", index=False)
    leagues.to_parquet(out_dir / "league_season_elo.parquet", index=False)


def read_tables(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = Path(processed_dir) / "elo"
    return (
        pd.read_parquet(base / "team_season_elo.parquet"),
        pd.read_parquet(base / "league_season_elo.parquet"),
    )
