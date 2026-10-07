"""Pure analysis functions behind scripts/feasibility.py, kept here so they are tested."""

from __future__ import annotations

from datetime import date

import pandas as pd

from scout.providers.api_football.adapter import NUMERIC_STATS

STAT_FIELDS = ["rating", *NUMERIC_STATS]
VALUE_STALENESS_DAYS = 365  # a valuation older than this does not describe "now"


def feature_date(season: int) -> pd.Timestamp:
    """Season end: season 2022 (2022/23) has feature date 2023-06-30."""
    return pd.Timestamp(date(season + 1, 6, 30))


def field_completeness(stats: pd.DataFrame) -> pd.DataFrame:
    """Share of non null values per field, by league and position, for players with minutes."""
    played = stats[stats["minutes"].fillna(0) > 0]
    grouped = played.groupby(["league_key", "position"], dropna=False)[STAT_FIELDS]
    table = grouped.agg(lambda s: s.notna().mean()).round(3)
    table["rows"] = played.groupby(["league_key", "position"], dropna=False).size()
    return table.reset_index()


def latest_value_asof(valuations: pd.DataFrame, players: pd.Series, when: pd.Timestamp):
    """Latest valuation per player on or before ``when``, if not older than a year."""
    window = valuations[
        valuations["source_player_id"].isin(players)
        & (valuations["valuation_date"] <= when)
        & (valuations["valuation_date"] > when - pd.Timedelta(days=VALUE_STALENESS_DAYS))
    ]
    return window.sort_values("valuation_date").groupby("source_player_id").tail(1)


def valuation_horizon_coverage(
    appearances: pd.DataFrame,
    valuations: pd.DataFrame,
    horizons_months: tuple[int, ...] = (24, 36),
    data_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """For each league and season: how many players have a value at the feature date
    and again ``h`` months later. That is the label coverage for the growth model."""
    last_value = data_end if data_end is not None else valuations["valuation_date"].max()
    rows = []
    for (league, season), group in appearances.groupby(["league_key", "season"]):
        players = group["source_player_id"].drop_duplicates()
        t0 = feature_date(int(season))
        at_t0 = latest_value_asof(valuations, players, t0)
        row = {
            "league": league,
            "season": int(season),
            "feature_date": t0.date().isoformat(),
            "players": len(players),
            "value_at_feature_date": len(at_t0),
        }
        for h in horizons_months:
            target = t0 + pd.DateOffset(months=h)
            if target > last_value:
                row[f"value_at_+{h}m"] = None  # horizon lies past the end of the data
                continue
            later = latest_value_asof(valuations, at_t0["source_player_id"], target)
            row[f"value_at_+{h}m"] = len(later)
        rows.append(row)
    return pd.DataFrame(rows)


def estimate_requests(
    pages_per_season: dict[tuple[str, int], int],
    seasons_per_league: dict[str, list[int]],
    fixtures_per_season: dict[str, int],
) -> pd.DataFrame:
    """Backfill request estimate per league.

    Season level stats need one request per ``/players`` page. Match level
    stats would need one ``/fixtures/players`` request per fixture plus one
    ``/fixtures`` list per season. Unknown page counts use the league's mean.
    """
    rows = []
    for league, seasons in seasons_per_league.items():
        known = [n for (lg, _), n in pages_per_season.items() if lg == league]
        mean_pages = sum(known) / len(known) if known else None
        season_pages = sum(pages_per_season.get((league, s), mean_pages or 0) for s in seasons)
        fixtures = fixtures_per_season.get(league, 0) * len(seasons)
        rows.append(
            {
                "league": league,
                "seasons": len(seasons),
                "pages_measured": len(known),
                "mean_pages_per_season": round(mean_pages, 1) if mean_pages else None,
                "season_stats_requests": round(season_pages),
                "match_stats_requests": fixtures + len(seasons),
            }
        )
    return pd.DataFrame(rows)
