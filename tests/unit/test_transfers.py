from pathlib import Path

import pandas as pd
import pytest
import yaml

from scout.backfill.transfers import club_ids, run_transfer_backfill, transfers_path
from scout.config import Settings
from scout.contracts import TRANSFERS, validate
from scout.providers.api_football.transfers import (
    parse_fee_eur,
    parse_transfers,
    transfer_kind,
    transfer_rows_to_frame,
)
from scout.ratelimit import DailyQuotaExhaustedError

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def settings(tmp_path):
    raw = yaml.safe_load((ROOT / "configs" / "backfill.ultra.yaml").read_text(encoding="utf-8"))
    raw["paths"] = {
        k: str(tmp_path / k) for k in ("raw_dir", "processed_dir", "state_dir", "reports_dir")
    }
    return Settings.model_validate(raw)


def move(date, out_id, in_id, kind="€ 25M"):
    return {
        "date": date,
        "type": kind,
        "teams": {
            "out": {"id": out_id, "name": f"Club {out_id}"},
            "in": {"id": in_id, "name": f"Club {in_id}"},
        },
    }


def payload(*players):
    return {
        "response": [
            {"player": {"id": pid, "name": f"P {pid}"}, "update": "2026-01-01", "transfers": moves}
            for pid, moves in players
        ]
    }


@pytest.mark.parametrize(
    ("raw", "fee", "kind"),
    [
        ("€ 25M", 25e6, "transfer"),
        ("€ 750K", 750e3, "transfer"),
        ("€ 2.5M", 2.5e6, "transfer"),
        ("35.4M €", 35.4e6, "transfer"),
        ("800K €", 800e3, "transfer"),
        ("6.5M", None, "transfer"),  # amount without a currency
        ("$ 3M", None, "transfer"),
        ("Transfer", None, "transfer"),
        ("Loan", None, "loan"),
        ("Back from Loan", None, "loan_return"),
        ("Return from loan", None, "loan_return"),
        ("End of Loan", None, "loan_return"),
        ("Free", None, "free"),
        ("Free agent", None, "free"),
        ("Free Transfer", None, "free"),
        ("Free €", None, "free"),
        ("Raise", None, "other"),
        ("Swap", None, "other"),
        ("End of career", None, "other"),
        ("N/A", None, "unknown"),
        ("-", None, "unknown"),
        (None, None, "unknown"),
    ],
)
def test_fee_and_kind_are_read_from_the_free_text_type(raw, fee, kind):
    assert parse_fee_eur(raw) == fee
    assert transfer_kind(raw) == kind


def test_the_same_move_seen_from_two_clubs_is_kept_once():
    from_seller = payload((7, [move("2019-07-01", 1, 2), move("2021-01-15", 2, 3, "Loan")]))
    from_buyer = payload((7, [move("2019-07-01", 1, 2)]))
    rows = parse_transfers(from_seller) + parse_transfers(from_buyer)
    frame = transfer_rows_to_frame(rows)
    validate(TRANSFERS, frame)
    assert len(frame) == 2
    assert frame["kind"].tolist() == ["transfer", "loan"]
    assert frame.loc[0, "fee_eur"] == 25e6


def test_rows_without_player_or_readable_date_are_dropped_and_unknown_clubs_missing(caplog):
    rows = parse_transfers(
        payload(
            (7, [move("2019-07-01", 0, 2), move("not a date", 1, 2), move("1926-01-20", 3, 4)]),
            (None, [move("2020-07-01", 1, 2)]),
        )
    )
    frame = transfer_rows_to_frame(rows)
    validate(TRANSFERS, frame)
    assert len(frame) == 1
    assert pd.isna(frame.loc[0, "team_out_id"])
    assert any("Dropping" in m for m in caplog.messages)


def test_the_same_move_listed_again_within_weeks_is_kept_once():
    rows = parse_transfers(
        payload(
            (
                7,
                [
                    move("2025-06-03", 1, 2, "Transfer"),
                    move("2025-06-30", 1, 2, "Transfer"),
                    move("2027-07-01", 1, 2, "Transfer"),  # a real second move back
                ],
            )
        )
    )
    frame = transfer_rows_to_frame(rows)
    assert frame["transfer_date"].dt.strftime("%Y-%m-%d").tolist() == ["2025-06-03", "2027-07-01"]


class FakeClient:
    requests_made = 0


class FakeProvider:
    def __init__(self, fail_after=None):
        self.client = FakeClient()
        self.fail_after = fail_after
        self.asked = []

    def team_transfers(self, team_id):
        if self.fail_after is not None and len(self.asked) >= self.fail_after:
            raise DailyQuotaExhaustedError("budget used")
        self.asked.append(team_id)
        self.client.requests_made += 1
        return payload((team_id * 100, [move("2020-07-01", team_id, team_id + 1)]))


def _season_stats(settings, league_key, season, teams):
    path = (
        settings.paths.processed_dir
        / "player_season_stats"
        / f"league={league_key}"
        / f"season={season}.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"team_id": pd.array(teams, dtype="Int64")}).to_parquet(path)


def test_one_request_per_club_across_leagues_and_seasons(settings):
    _season_stats(settings, "epl", 2020, [1, 2, None])
    _season_stats(settings, "epl", 2021, [2, 3])
    _season_stats(settings, "super_lig", 2020, [4])
    assert club_ids(settings) == [1, 2, 3, 4]
    assert club_ids(settings, ["super_lig"]) == [4]

    provider = FakeProvider()
    result = run_transfer_backfill(settings, provider)
    assert result.complete
    assert provider.asked == [1, 2, 3, 4]
    assert result.requests_made == 4
    written = pd.read_parquet(transfers_path(settings))
    assert len(written) == 4


def test_nothing_is_written_until_every_club_is_cached(settings):
    _season_stats(settings, "epl", 2020, [1, 2, 3])
    result = run_transfer_backfill(settings, FakeProvider(fail_after=2))
    assert result.status == "partial"
    assert result.clubs_done == 2
    assert result.stopped_reason.startswith("quota")
    assert not transfers_path(settings).exists()
