"""``/transfers``: the transfer history of every player a club has had.

One request per club returns, for each of its players, that player's whole
career of moves (not only the moves involving this club), so the same
transfer arrives once per club the player passed through. Rows are
deduplicated on player, date and both clubs.

``type`` is free text: a fee ("€ 25M", "€ 750K"), "Loan", "Back from Loan",
"Free", "N/A" or "-". It is kept as sent in ``type_raw`` and read into
``kind`` and, for euro fees, ``fee_eur``.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

from scout.providers.api_football.client import SOURCE, ApiFootballClient

log = logging.getLogger(__name__)

TRANSFER_KINDS = ["fee", "loan", "loan_return", "free", "unknown"]
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

_FEE = re.compile(r"^\s*€\s*([0-9]+(?:[.,][0-9]+)?)\s*([MK]?)\s*$", re.IGNORECASE)


class ApiFootballTransferProvider:
    def __init__(self, client: ApiFootballClient) -> None:
        self.client = client

    def team_transfers(self, team_id: int) -> dict[str, Any]:
        return self.client.get("transfers", {"team": team_id}).payload


def parse_fee_eur(type_raw: str | None) -> float | None:
    """'€ 25M' -> 25_000_000.0; anything that is not a euro amount -> None."""
    match = _FEE.match(type_raw or "")
    if not match:
        return None
    amount = float(match.group(1).replace(",", "."))
    unit = match.group(2).upper()
    return amount * {"M": 1e6, "K": 1e3, "": 1.0}[unit]


def transfer_kind(type_raw: str | None) -> str:
    text = (type_raw or "").strip().lower()
    if "loan" in text:
        return "loan_return" if ("back" in text or "return" in text or "end" in text) else "loan"
    if text.startswith("free"):
        return "free"
    if parse_fee_eur(type_raw) is not None or re.match(r"^[€$£]\s*[0-9]", text):
        return "fee"
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

    * rows without a player id, or with a date that can't be read, are dropped
    * a club id of 0 or below means "unknown club" and becomes missing
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
    no_date = frame["transfer_date"].isna()
    if no_id.any() or no_date.any():
        log.warning(
            "Dropping %d transfers without a player id and %d without a readable date",
            int(no_id.sum()),
            int((no_date & ~no_id).sum()),
        )
    frame = frame[~(no_id | no_date)]
    frame = frame.drop_duplicates(TRANSFER_KEY, keep="first")
    return frame.sort_values(TRANSFER_KEY[:2], kind="stable").reset_index(drop=True)
