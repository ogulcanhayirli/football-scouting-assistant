"""Pure analysis functions behind scripts/feasibility.py, kept here so they are tested."""

from __future__ import annotations

from datetime import date

import pandas as pd

from scout.providers.api_football.adapter import NUMERIC_STATS

STAT_FIELDS = list(NUMERIC_STATS)
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


# Fields that decide which models are possible: rating and the per action
# counts carry most of the signal beyond minutes and goals.
KEY_FIELDS = [
    "minutes",
    "goals",
    "assists",
    "rating",
    "shots_total",
    "passes_total",
    "passes_key",
    "tackles_total",
    "duels_total",
    "dribbles_attempts",
]
MIN_MINUTES = 450  # five full matches; below that a season says little about a player
USABLE_SHARE = 0.9


def completeness_by_season(
    stats: pd.DataFrame, fields: list[str] = KEY_FIELDS, min_minutes: int = MIN_MINUTES
) -> pd.DataFrame:
    """Share of non null values per field, by league and season, for regular players."""
    played = stats[stats["minutes"].fillna(0) >= min_minutes]
    grouped = played.groupby(["league_key", "season"])
    table = grouped[fields].agg(lambda s: s.notna().mean()).round(3)
    table.insert(0, "players", grouped.size())
    return table.reset_index()


def first_usable_season(table: pd.DataFrame, threshold: float = USABLE_SHARE) -> pd.DataFrame:
    """Per league and field, the first season from which every later season is at least
    ``threshold`` complete. None when even the newest season falls short."""
    fields = [c for c in table.columns if c not in {"league_key", "season", "players"}]
    rows = []
    for league, group in table.sort_values("season").groupby("league_key"):
        row: dict = {"league": league}
        for field in fields:
            first = None
            for season, share in zip(group["season"], group[field], strict=True):
                if share >= threshold:
                    first = first if first is not None else int(season)
                else:
                    first = None
            row[field] = first
        rows.append(row)
    return pd.DataFrame(rows)


def label_coverage(
    stats: pd.DataFrame,
    matches: pd.DataFrame,
    valuations: pd.DataFrame,
    min_minutes: int = MIN_MINUTES,
) -> pd.DataFrame:
    """Per league and season: regular players, how many are matched to Transfermarkt,
    and how many of those have a value at the season's feature date. That last number
    is the training set size for the fair value model."""
    regular = stats[stats["minutes"].fillna(0) >= min_minutes]
    regular = regular.drop_duplicates(["league_key", "season", "source_player_id"])
    # A player who played in both leagues is matched once per league; keep one link.
    links = matches.loc[matches["reason"] == "matched", ["af_player_id", "tm_player_id"]]
    links = links.drop_duplicates("af_player_id")
    linked = regular.merge(
        links,
        left_on="source_player_id",
        right_on="af_player_id",
        how="left",
    )
    rows = []
    for (league, season), group in linked.groupby(["league_key", "season"]):
        matched = group["tm_player_id"].dropna()
        valued = latest_value_asof(valuations, matched, feature_date(int(season)))
        rows.append(
            {
                "league": league,
                "season": int(season),
                "regular_players": len(group),
                "matched": len(matched),
                "match_rate": round(len(matched) / len(group), 3) if len(group) else None,
                "valued_at_feature_date": len(valued),
                "label_rate": round(len(valued) / len(group), 3) if len(group) else None,
            }
        )
    return pd.DataFrame(rows)


def conflicting_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Stats players matched to different Transfermarkt players in different leagues."""
    links = matches.loc[matches["reason"] == "matched", ["af_player_id", "tm_player_id"]]
    counts = links.drop_duplicates().groupby("af_player_id")["tm_player_id"].nunique()
    return links[links["af_player_id"].isin(counts[counts > 1].index)].drop_duplicates()


def value_vs_fee(
    transfers: pd.DataFrame,
    valuations: pd.DataFrame,
    club_leagues: pd.DataFrame,
    data_end: pd.Timestamp,
) -> pd.DataFrame:
    """Paid transfers into or out of our leagues, each with the player's latest
    Transfermarkt value on or before the transfer date (at most a year old).

    ``club_leagues`` has club_id and league_key for clubs that played in a league.
    A transfer between two clubs of the same league appears once for that league.
    """
    paid = transfers[
        (transfers["transfer_fee_eur"] > 0)
        & transfers["transfer_date"].notna()
        & (transfers["transfer_date"] <= data_end)
    ]
    sides = []
    for column, direction in (("to_club_id", "in"), ("from_club_id", "out")):
        side = paid.merge(
            club_leagues.drop_duplicates(), left_on=column, right_on="club_id", how="inner"
        )
        sides.append(side.assign(direction=direction))
    joined = pd.concat(sides, ignore_index=True).drop_duplicates(
        ["source_player_id", "transfer_date", "league_key"]
    )
    vals = valuations.dropna(subset=["valuation_date", "market_value_eur"])
    vals = vals[vals["market_value_eur"] > 0]
    merged = pd.merge_asof(
        joined.sort_values("transfer_date"),
        vals[["source_player_id", "valuation_date", "market_value_eur"]].sort_values(
            "valuation_date"
        ),
        left_on="transfer_date",
        right_on="valuation_date",
        by="source_player_id",
        direction="backward",
        tolerance=pd.Timedelta(days=VALUE_STALENESS_DAYS),
    )
    merged = merged.dropna(subset=["market_value_eur"])
    merged["fee_to_value"] = merged["transfer_fee_eur"] / merged["market_value_eur"]
    return merged[
        [
            "league_key",
            "direction",
            "source_player_id",
            "transfer_date",
            "transfer_fee_eur",
            "market_value_eur",
            "fee_to_value",
        ]
    ].reset_index(drop=True)


def _spearman(a: pd.Series, b: pd.Series) -> float | None:
    """Pearson correlation of ranks (pandas' own Spearman needs scipy). None when
    there are fewer than three pairs, where a correlation means nothing."""
    if len(a) < 3:
        return None
    return round(float(a.rank().corr(b.rank())), 3)


def summarize_value_vs_fee(pairs: pd.DataFrame) -> pd.DataFrame:
    """How well the value before a transfer predicts the fee, per league.

    Spearman correlation is rank based, so it needs no log transform.
    ``within_2x`` is the share of fees between half and double the value.
    """
    rows = []
    for league, group in pairs.groupby("league_key"):
        ratio = group["fee_to_value"]
        rows.append(
            {
                "league": league,
                "paid_transfers": len(group),
                "spearman": _spearman(group["market_value_eur"], group["transfer_fee_eur"]),
                "median_fee_to_value": round(float(ratio.median()), 2),
                "within_2x": round(float(((ratio >= 0.5) & (ratio <= 2)).mean()), 3),
                "first_transfer": group["transfer_date"].min().date().isoformat(),
            }
        )
    return pd.DataFrame(rows)
