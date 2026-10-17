"""transfermarkt-datasets adapter: snapshot CSVs to canonical players and valuations.

The dataset (github.com/dcaribou/transfermarkt-datasets) is published as
gzipped CSVs. It has no stated license, so we use it for non commercial work
only and attribute it. Updates paused in July 2026; valuations end on
2026-06-12, which bounds the backtest window. ``data_end`` reads that date
from the snapshot, so a refreshed snapshot moves the window by itself.

``refresh_snapshot`` checks the published files against the cached ones and
downloads only those that changed; the files they replace are moved to
``archive/<date>/``, never deleted.

Players are scoped to a league through ``appearances``: anyone who played a
domestic league match there. That is more complete than the current club
field, which only says where a player is now.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

from scout.config import LeagueConfig, TransfermarktConfig
from scout.fsutil import atomic_write_text

log = logging.getLogger(__name__)

SOURCE = "transfermarkt"

REQUIRED_COLUMNS: dict[str, set[str]] = {
    "players.csv.gz": {
        "player_id",
        "name",
        "first_name",
        "last_name",
        "date_of_birth",
        "position",
        "sub_position",
        "country_of_citizenship",
        "current_club_id",
        "current_club_name",
        "contract_expiration_date",
        "agent_name",
        "highest_market_value_in_eur",
    },
    "player_valuations.csv.gz": {
        "player_id",
        "date",
        "market_value_in_eur",
        "current_club_id",
        "player_club_domestic_competition_id",
    },
    "transfers.csv.gz": {
        "player_id",
        "transfer_date",
        "from_club_id",
        "to_club_id",
        "transfer_fee",
    },
    "games.csv.gz": {"game_id", "competition_id", "season", "date"},
    "appearances.csv.gz": {
        "player_id",
        "game_id",
        "player_club_id",
        "date",
        "competition_id",
        "minutes_played",
    },
}


def season_of(dates: pd.Series) -> pd.Series:
    """European season start year: July 2023 to June 2024 is season 2023."""
    dates = pd.to_datetime(dates)
    return (dates.dt.year - (dates.dt.month < 7).astype(int)).astype("Int64")


# A contract running outside these years is an entry error, not a contract.
CONTRACT_RANGE = (pd.Timestamp("2000-01-01"), pd.Timestamp("2040-12-31"))
MAX_MARKET_VALUE_EUR = 500_000_000

KEPT_HEADERS = {"etag", "last-modified", "content-length"}
ARCHIVE_DIR = "archive"


def _kept_headers(headers: httpx.Headers) -> dict[str, str]:
    return {k.lower(): v for k, v in headers.items() if k.lower() in KEPT_HEADERS}


def _fetch(http: httpx.Client, url: str, path: Path) -> dict[str, str]:
    """Stream ``url`` to ``path``: the body lands through a rename, the sidecar first."""
    tmp = path.with_suffix(path.suffix + ".part")
    with http.stream("GET", url) as response:
        response.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in response.iter_bytes():
                fh.write(chunk)
        headers = _kept_headers(response.headers)
    meta = {
        "url": url,
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "headers": headers,
    }
    atomic_write_text(path.with_name(path.name + ".meta.json"), json.dumps(meta, indent=2))
    tmp.replace(path)
    return headers


def _url(config: TransfermarktConfig, name: str) -> str:
    return f"{config.base_url.rstrip('/')}/{name}"


def download_snapshot(
    config: TransfermarktConfig,
    raw_dir: Path,
    *,
    transport: httpx.BaseTransport | None = None,
) -> list[Path]:
    """Download each snapshot file once. Existing files are kept untouched."""
    target = Path(raw_dir) / SOURCE
    target.mkdir(parents=True, exist_ok=True)
    paths = []
    with httpx.Client(timeout=300, transport=transport, follow_redirects=True) as http:
        for name in config.files:
            path = target / name
            paths.append(path)
            if path.is_file():
                log.info("transfermarkt %s cached, skipping", name)
                continue
            _fetch(http, _url(config, name), path)
            log.info("transfermarkt %s downloaded", name)
    return paths


@dataclass(frozen=True)
class FileRefresh:
    name: str
    status: str  # "unchanged", "new" or "updated"
    last_modified: str | None
    archived_to: Path | None = None


def _same_release(cached: dict[str, str], remote: dict[str, str]) -> bool:
    """The ETag decides; without one, size and modification time must both match."""
    if cached.get("etag") and remote.get("etag"):
        return cached["etag"] == remote["etag"]
    keys = ("content-length", "last-modified")
    return all(cached.get(k) and cached.get(k) == remote.get(k) for k in keys)


def _archive(path: Path, cached: dict[str, str], fetched_at: str) -> Path:
    """Move a superseded file and its sidecar to ``archive/<date>/``; nothing is deleted."""
    stamp = cached.get("last-modified")
    day = pd.Timestamp(stamp).date() if stamp else pd.Timestamp(fetched_at).date()
    folder = path.parent / ARCHIVE_DIR / day.isoformat()
    folder.mkdir(parents=True, exist_ok=True)
    meta = path.with_name(path.name + ".meta.json")
    if (folder / path.name).exists():
        raise FileExistsError(f"{folder / path.name} already archived; refusing to overwrite")
    if meta.is_file():
        meta.replace(folder / meta.name)
    path.replace(folder / path.name)
    return folder


def refresh_snapshot(
    config: TransfermarktConfig,
    raw_dir: Path,
    *,
    transport: httpx.BaseTransport | None = None,
) -> list[FileRefresh]:
    """Bring the cached snapshot up to the latest published release.

    A HEAD request per file compares the published ETag with the one in the
    cached sidecar. Unchanged files are not downloaded again; a changed file
    is downloaded and the old copy moved to ``archive/<last modified date>/``.
    """
    target = Path(raw_dir) / SOURCE
    target.mkdir(parents=True, exist_ok=True)
    out = []
    with httpx.Client(timeout=300, transport=transport, follow_redirects=True) as http:
        for name in config.files:
            path = target / name
            url = _url(config, name)
            head = http.head(url)
            head.raise_for_status()
            remote = _kept_headers(head.headers)
            meta_path = path.with_name(name + ".meta.json")
            meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
            cached = {k.lower(): v for k, v in meta.get("headers", {}).items()}
            if path.is_file() and _same_release(cached, remote):
                log.info("transfermarkt %s unchanged (%s)", name, remote.get("last-modified"))
                out.append(FileRefresh(name, "unchanged", remote.get("last-modified")))
                continue
            archived = None
            if path.is_file():
                archived = _archive(path, cached, meta.get("fetched_at", ""))
            headers = _fetch(http, url, path)
            status = "updated" if archived else "new"
            log.info("transfermarkt %s %s (%s)", name, status, headers.get("last-modified"))
            out.append(FileRefresh(name, status, headers.get("last-modified"), archived))
    return out


class TransfermarktValuationProvider:
    source = SOURCE

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self._cache: dict[str, pd.DataFrame] = {}

    def _read(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            frame = pd.read_csv(self.data_dir / name, low_memory=False)
            missing = REQUIRED_COLUMNS.get(name, set()) - set(frame.columns)
            if missing:
                raise ValueError(f"{name} is missing columns {sorted(missing)}; schema changed?")
            self._cache[name] = frame
        return self._cache[name]

    def _game_seasons(self, apps: pd.DataFrame) -> pd.Series:
        """Season of each appearance, from the dataset's games table when present.

        A 1 July cut off is wrong for 2019/20: the COVID restart ran into late
        July 2020, and 111 EPL and Süper Lig matches would land in 2020/21. The
        date rule is only the fallback for games the table does not list.
        """
        by_date = season_of(apps["date"])
        if not (self.data_dir / "games.csv.gz").is_file():
            log.warning("games.csv.gz missing; seasons fall back to a 1 July cut off")
            return by_date
        games = self._read("games.csv.gz")
        seasons = games.set_index("game_id")["season"]
        from_games = apps["game_id"].map(seasons).astype("Int64")
        return from_games.fillna(by_date)

    def league_appearances(self, leagues: list[LeagueConfig]) -> pd.DataFrame:
        """Minutes per player, club, league and season, from match appearances."""
        codes = {lg.transfermarkt_id: lg.key for lg in leagues}
        apps = self._read("appearances.csv.gz")
        apps = apps[apps["competition_id"].isin(codes)].copy()
        apps["season"] = self._game_seasons(apps)
        apps["league_key"] = apps["competition_id"].map(codes)
        grouped = (
            apps.groupby(["player_id", "player_club_id", "league_key", "season"], as_index=False)
            .agg(matches=("date", "size"), minutes=("minutes_played", "sum"))
            .rename(columns={"player_id": "source_player_id", "player_club_id": "club_id"})
        )
        return grouped

    def players(self, leagues: list[LeagueConfig] | None) -> pd.DataFrame:
        """Players who appeared in ``leagues``, or every player when ``leagues`` is None."""
        players = self._read("players.csv.gz")
        if leagues is not None:
            ids = self.league_appearances(leagues)["source_player_id"].unique()
            players = players[players["player_id"].isin(ids)]
        out = pd.DataFrame(
            {
                "source": SOURCE,
                "source_player_id": players["player_id"].astype("Int64"),
                "player_name": players["name"],
                "first_name": players["first_name"],
                "last_name": players["last_name"],
                "birth_date": pd.to_datetime(players["date_of_birth"], errors="coerce"),
                "position": players["position"],
                "sub_position": players["sub_position"],
                "nationality": players["country_of_citizenship"],
                "current_club_id": players["current_club_id"].astype("Int64"),
                "current_club_name": players["current_club_name"],
                "foot": players.get("foot"),
                "height_cm": pd.to_numeric(players.get("height_in_cm"), errors="coerce"),
            }
        )
        return out.reset_index(drop=True)

    def data_end(self) -> pd.Timestamp:
        """Date of the last valuation in the snapshot: labels after it are missing."""
        dates = pd.to_datetime(self._read("player_valuations.csv.gz")["date"], errors="coerce")
        return dates.max().normalize()

    def profiles(self) -> pd.DataFrame:
        """Contract expiry, agent and highest market value, as the snapshot states them.

        These describe the player on the snapshot date, not at any feature date,
        so they are for display in the current season's lists only and must
        never become model inputs. An impossible contract year or value is an
        entry error and becomes missing.
        """
        players = self._read("players.csv.gz")
        contract = pd.to_datetime(players["contract_expiration_date"], errors="coerce")
        bad_contract = contract.notna() & ~contract.between(*CONTRACT_RANGE)
        highest = pd.to_numeric(players["highest_market_value_in_eur"], errors="coerce")
        bad_value = highest.notna() & ~highest.between(0, MAX_MARKET_VALUE_EUR, inclusive="right")
        for bad, what in ((bad_contract, "contract expiry"), (bad_value, "highest value")):
            if bad.any():
                log.warning("transfermarkt: %d impossible %s stored as missing", bad.sum(), what)
        agent = players["agent_name"].astype("string").str.strip()
        return pd.DataFrame(
            {
                "source_player_id": players["player_id"].astype("Int64"),
                "contract_expiry": contract.mask(bad_contract),
                "agent": agent.mask(agent == ""),
                "highest_market_value_eur": highest.mask(bad_value).astype(float),
            }
        ).reset_index(drop=True)

    def valuations(self, leagues: list[LeagueConfig] | None) -> pd.DataFrame:
        """Valuations of players who appeared in ``leagues``, or of everyone when None."""
        vals = self._read("player_valuations.csv.gz")
        if leagues is not None:
            ids = self.league_appearances(leagues)["source_player_id"].unique()
            vals = vals[vals["player_id"].isin(ids)]
        out = pd.DataFrame(
            {
                "source": SOURCE,
                "source_player_id": vals["player_id"].astype("Int64"),
                "valuation_date": pd.to_datetime(vals["date"], errors="coerce"),
                "market_value_eur": pd.to_numeric(
                    vals["market_value_in_eur"], errors="coerce"
                ).astype(float),
                "club_id": vals["current_club_id"].astype("Int64"),
                "club_competition_id": vals["player_club_domestic_competition_id"],
            }
        )
        return out.sort_values(["source_player_id", "valuation_date"]).reset_index(drop=True)

    def transfers(self) -> pd.DataFrame:
        """Transfers with a known fee. A fee of 0 is a free move or a loan, so it is kept
        as 0; a blank fee is unknown and becomes missing."""
        raw = self._read("transfers.csv.gz")
        return pd.DataFrame(
            {
                "source_player_id": raw["player_id"].astype("Int64"),
                "transfer_date": pd.to_datetime(raw["transfer_date"], errors="coerce"),
                "from_club_id": pd.to_numeric(raw["from_club_id"], errors="coerce").astype("Int64"),
                "to_club_id": pd.to_numeric(raw["to_club_id"], errors="coerce").astype("Int64"),
                "transfer_fee_eur": pd.to_numeric(raw["transfer_fee"], errors="coerce"),
            }
        )
