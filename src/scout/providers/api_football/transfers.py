"""``/transfers``: the transfer history of every player a club has had.

One request per club returns, for each of its players, that player's whole
career of moves (not only the moves involving this club), so the same
transfer arrives once per club the player passed through. Rows are
deduplicated on player, date and both clubs.

``type`` is free text. Seen in real responses: fees written "€ 25M",
"25M €" or "6.5M"; "Transfer" (a permanent move, fee not given); "Loan";
"Back from Loan", "Return from loan", "End of Loan"; "Free", "Free agent",
"Free Transfer", "Free €"; "Raise" (from the youth team), "Swap",
"End of career"; "N/A" and "-". It is kept as sent in ``type_raw`` and read
into ``kind`` and, for fees in euros, ``fee_eur``.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

from scout.providers.api_football.client import SOURCE, ApiFootballClient

log = logging.getLogger(__name__)

TRANSFER_KINDS = ["transfer", "loan", "loan_return", "free", "other", "unknown"]
OTHER_TYPES = {"raise", "swap", "end of career"}
# Some moves carry a placeholder year (1926 for moves made in 2024 and 2025).
EARLIEST_DATE = pd.Timestamp("1950-01-01")
# The same move listed twice a few weeks apart (C. Kelleher, 3 and 30 June 2025).
REPEAT_DAYS = 60
TRANSFER_COLUMNS = [
    "source",
    "source_player_id",
    "player_name",
    "transfer_date",
    "team_out_id",
    "team_out_name",
    "team_in_id",
    "team_in_name",
    "type_raw",
    "kind",
    "fee_eur",
]
TRANSFER_KEY = ["source_player_id", "transfer_date", "team_out_id", "team_in_id"]

_AMOUNT = r"([0-9]+(?:[.,][0-9]+)?)\s*([MK]?)"
_FEE_EUR = re.compile(rf"^\s*(?:€\s*{_AMOUNT}|{_AMOUNT}\s*€)\s*$", re.IGNORECASE)
_FEE_ANY = re.compile(r"^\s*[€$£]?\s*[0-9]+(?:[.,][0-9]+)?\s*[MK]?\s*[€$£]?\s*$", re.IGNORECASE)


class ApiFootballTransferProvider:
    def __init__(self, client: ApiFootballClient) -> None:
        self.client = client

    def team_transfers(self, team_id: int) -> dict[str, Any]:
        return self.client.get("transfers", {"team": team_id}).payload


def parse_fee_eur(type_raw: str | None) -> float | None:
    """'€ 25M' or '25M €' -> 25_000_000.0; anything not marked as euros -> None."""
    match = _FEE_EUR.match(type_raw or "")
    if not match:
        return None
    groups = [g for g in match.groups() if g is not None]
    amount, unit = float(groups[0].replace(",", ".")), (groups[1] if len(groups) > 1 else "")
    return amount * {"M": 1e6, "K": 1e3, "": 1.0}[unit.upper()]


def transfer_kind(type_raw: str | None) -> str:
    text = (type_raw or "").strip().lower()
    if "loan" in text:
        return "loan_return" if ("back" in text or "return" in text or "end" in text) else "loan"
    if text.startswith("free"):
        return "free"
    if text == "transfer" or _FEE_ANY.match(text):
        return "transfer"
    if text in OTHER_TYPES:
        return "other"
    return "unknown"


def parse_transfers(payload: dict[str, Any]) -> list[dict]:
    rows = []
    for item in payload.get("response") or []:
        player = item.get("player") or {}
        for move in item.get("transfers") or []:
            teams = move.get("teams") or {}
            team_in = teams.get("in") or {}
            team_out = teams.get("out") or {}
            type_raw = move.get("type")
            rows.append(
                {
                    "source": SOURCE,
                    "source_player_id": player.get("id"),
                    "player_name": player.get("name"),
                    "transfer_date": move.get("date"),
                    "team_out_id": team_out.get("id"),
                    "team_out_name": team_out.get("name"),
                    "team_in_id": team_in.get("id"),
                    "team_in_name": team_in.get("name"),
                    "type_raw": type_raw,
                    "kind": transfer_kind(type_raw),
                    "fee_eur": parse_fee_eur(type_raw),
                }
            )
    return rows


def transfer_rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    """Typed, deduplicated transfers. Source errors are logged and repaired:

    * rows without a player id, or with a date that can't be read or lies
      before 1950 (a placeholder year), are dropped
    * a club id of 0 or below means "unknown club" and becomes missing
    * the same move (player and both clubs) listed again within 60 days is
      kept once, at its first date
    """
    frame = pd.DataFrame(rows, columns=TRANSFER_COLUMNS)
    for column in ["source_player_id", "team_out_id", "team_in_id"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    for column in ["team_out_id", "team_in_id"]:
        frame.loc[(frame[column] <= 0).fillna(False), column] = pd.NA
    frame["fee_eur"] = pd.to_numeric(frame["fee_eur"], errors="coerce").astype("Float64")
    frame["transfer_date"] = pd.to_datetime(
        frame["transfer_date"], errors="coerce", format="%Y-%m-%d"
    ).astype("datetime64[ns]")
    no_id = frame["source_player_id"].isna() | (frame["source_player_id"] <= 0).fillna(True)
    no_date = frame["transfer_date"].isna() | (frame["transfer_date"] < EARLIEST_DATE)
    if no_id.any() or no_date.any():
        log.warning(
            "Dropping %d transfers without a player id and %d without a usable date",
            int(no_id.sum()),
            int((no_date & ~no_id).sum()),
        )
    frame = frame[~(no_id | no_date)]
    frame = frame.drop_duplicates(TRANSFER_KEY, keep="first")
    frame = frame.sort_values(TRANSFER_KEY[:2], kind="stable")
    move = ["source_player_id", "team_out_id", "team_in_id"]
    gap = frame.groupby(move, dropna=False)["transfer_date"].diff()
    repeat = (gap <= pd.Timedelta(days=REPEAT_DAYS)).fillna(False)
    if repeat.any():
        log.warning("Dropping %d moves listed again within %d days", int(repeat.sum()), REPEAT_DAYS)
    return frame[~repeat].reset_index(drop=True)
