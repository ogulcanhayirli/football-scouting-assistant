from datetime import UTC, datetime

import pytest

from scout.ratelimit import DailyQuotaExhaustedError, QuotaGuard


class FakeClock:
    def __init__(self, start: float) -> None:
        self.now = start
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


NOON = datetime(2026, 10, 7, 12, 0, tzinfo=UTC).timestamp()


def make_guard(clock, tmp_path=None, **kw):
    params = {"daily_limit": 100, "per_minute_limit": 10, "safety_margin": 0} | kw
    state = tmp_path / "quota.json" if tmp_path else None
    return QuotaGuard(**params, state_path=state, clock=clock, sleep=clock.sleep)


def test_never_exceeds_per_minute_limit():
    clock = FakeClock(NOON)
    guard = make_guard(clock)
    times = []
    for _ in range(25):
        guard.acquire()
        times.append(clock.now)
    # In any 60 second span there are at most 10 requests.
    for i, t in enumerate(times):
        assert sum(1 for u in times[i:] if u - t < 60) <= 10
    assert clock.slept, "the guard should have waited"


def test_daily_budget_respects_safety_margin():
    clock = FakeClock(NOON)
    guard = make_guard(clock, daily_limit=12, per_minute_limit=100, safety_margin=2)
    for _ in range(10):
        guard.acquire()
    with pytest.raises(DailyQuotaExhaustedError):
        guard.acquire()


def test_daily_counter_survives_restart(tmp_path):
    clock = FakeClock(NOON)
    guard = make_guard(clock, tmp_path, daily_limit=5, per_minute_limit=100)
    for _ in range(3):
        guard.acquire()
    restarted = make_guard(clock, tmp_path, daily_limit=5, per_minute_limit=100)
    assert restarted.used_today == 3
    restarted.acquire()
    restarted.acquire()
    with pytest.raises(DailyQuotaExhaustedError):
        restarted.acquire()


def test_daily_counter_resets_at_utc_midnight(tmp_path):
    clock = FakeClock(NOON)
    guard = make_guard(clock, tmp_path, daily_limit=2, per_minute_limit=100)
    guard.acquire()
    guard.acquire()
    clock.now += 12 * 3600 + 1  # past 00:00 UTC
    guard.acquire()
    assert guard.used_today == 1


def test_server_headers_only_tighten_budget():
    clock = FakeClock(NOON)
    guard = make_guard(clock)
    guard.acquire()
    guard.observe_server_remaining(daily_remaining=40)  # someone else used 59
    assert guard.used_today == 60
    guard.observe_server_remaining(daily_remaining=99)  # stale header, ignored
    assert guard.used_today == 60


def test_server_minute_exhausted_forces_wait():
    clock = FakeClock(NOON)
    guard = make_guard(clock)
    guard.acquire()
    guard.observe_server_remaining(daily_remaining=None, minute_remaining=0)
    guard.acquire()
    assert clock.slept
    assert clock.slept[0] > 59
