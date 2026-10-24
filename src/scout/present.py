"""Today's club and the season so far, joined into the newest tables (prompt 16).

``scout lists`` calls ``add_present`` after building the season's tables:

* ``current_club`` and ``current_league`` from
  ``data/processed/current/current_club.parquet`` (``scout current``);
* ``moved_since_forecast``: the club today is not the club at forecast,
  compared by club id (the forecast's club is the row's ``team_id`` in
  ``scouting.parquet``);
* the ``so_far_*`` block from ``data/processed/current/season_so_far.parquet``
  (``scout features --current``), with ``so_far_minutes`` 0 for a player
  without minutes in our leagues this season.

Without those files the columns stay blank, so the tables keep their shape.
Display only: nothing here reaches a feature table or a model.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


def _read(path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.is_file() else None


def present_frame(
    players: pd.DataFrame,
    current: pd.DataFrame | None,
    so_far: pd.DataFrame | None,
    leagues: dict[str, str],
) -> pd.DataFrame:
    """The present columns for ``players`` (player_id, forecast_team_id), in order."""
    from scout.lists import PRESENT_COLUMNS

    out = pd.DataFrame(index=players.index)
    ids = players["player_id"].astype("Int64")
    if current is not None:
        cur = current.set_index("source_player_id")
        out["current_club"] = ids.map(cur["current_club"]).astype(object)
        out["current_league"] = ids.map(cur["current_league"]).astype(object)
        club_id = ids.map(cur["current_club_id"]).astype("Int64")
        forecast = players["forecast_team_id"].astype("Int64")
        moved = (club_id != forecast).astype("boolean")
        out["moved_since_forecast"] = moved.where(club_id.notna() & forecast.notna())
    if so_far is not None:
        block = so_far.set_index("source_player_id")
        for column in PRESENT_COLUMNS:
            if column in block:
                out[column] = ids.map(block[column])
        out["so_far_league"] = ids.map(block["so_far_league_key"]).map(leagues)
        out["so_far_date"] = so_far["so_far_date"].max()
        out["so_far_minutes"] = out["so_far_minutes"].fillna(0.0)
    for column in PRESENT_COLUMNS:
        if column not in out:
            out[column] = pd.NA
    out["moved_since_forecast"] = out["moved_since_forecast"].astype("boolean")
    out["so_far_date"] = pd.to_datetime(out["so_far_date"]).astype("datetime64[ns]")
    for column in ("so_far_matches", "so_far_flag_count"):
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")
    return out[PRESENT_COLUMNS]


def forecast_team_ids(table: pd.DataFrame, scouting: pd.DataFrame) -> pd.Series:
    """The ``team_id`` of each row's club at forecast (same player, season and club)."""
    rows = scouting[["season", "source_player_id", "team_name", "team_id"]].rename(
        columns={"source_player_id": "player_id", "team_name": "club_at_forecast"}
    )
    rows = rows.drop_duplicates(["season", "player_id", "club_at_forecast"])
    merged = table[["season", "player_id", "club_at_forecast"]].merge(
        rows.astype({"player_id": "Int64"}),
        on=["season", "player_id", "club_at_forecast"],
        how="left",
    )
    return pd.Series(merged["team_id"].to_numpy(), index=table.index)


def add_present(
    settings, cand: pd.DataFrame, barg: pd.DataFrame, scouting: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from scout.lists import DICTIONARY, league_names

    current_dir = settings.paths.processed_dir / "current"
    current = _read(current_dir / "current_club.parquet")
    so_far = _read(current_dir / "season_so_far.parquet")
    if current is None:
        log.warning("lists: no current_club table; run `scout current` (columns left blank)")
    if so_far is None:
        log.warning("lists: no season so far block; run `scout features --current`")
    leagues = league_names(settings)
    out = []
    for name, table in (("candidates", cand), ("bargains", barg)):
        players = pd.DataFrame(
            {
                "player_id": table["player_id"],
                "forecast_team_id": forecast_team_ids(table, scouting),
            }
        )
        extra = present_frame(players, current, so_far, leagues)
        both = pd.concat([table, extra], axis=1)[list(DICTIONARY[name])]
        out.append(both)
    cand, barg = out
    moved = cand["moved_since_forecast"].fillna(False).astype(bool)
    log.info(
        "lists: %d of %d candidates (%d bargains) moved since the forecast; %.1f%% of "
        "candidates have minutes this season",
        int(moved.sum()),
        len(cand),
        int(barg["moved_since_forecast"].fillna(False).astype(bool).sum()),
        100 * (cand["so_far_minutes"].astype(float) > 0).mean(),
    )
    return cand, barg
