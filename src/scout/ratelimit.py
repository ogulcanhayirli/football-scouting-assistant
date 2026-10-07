"""Quota guard for APIs with a per minute and a per day request limit.

Two limits are enforced before every request:

* a sliding 60 second window, which waits when the window is full;
* a daily counter persisted to disk, which resets at UTC midnight (API Football
  resets its daily quota at 00:00 UTC) and raises when the budget is spent.

Requests are counted before they are sent. If the process dies mid request we
overcount by one, which is the safe direction. The server's own remaining
quota headers can only lower our budget, never raise it.
"""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from scout.fsutil import atomic_write_text

WINDOW_SECONDS = 60.0


class DailyQuotaExhaustedError(RuntimeError):
    pass


def _utc_today(now: float) -> str:
    return datetime.fromtimestamp(now, tz=UTC).date().isoformat()


class QuotaGuard:
    def __init__(
        self,
        *,
        daily_limit: int,
        per_minute_limit: int,
        safety_margin: int = 0,
        state_path: Path | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.daily_limit = daily_limit
        self.per_minute_limit = per_minute_limit
        self.safety_margin = safety_margin
        self.state_path = state_path
        self._clock = clock
        self._sleep = sleep
        self._window: deque[float] = deque()
        self._day, self._used_today = self._load_state()

    # Daily budget

    @property
    def daily_budget(self) -> int:
        return self.daily_limit - self.safety_margin

    @property
    def used_today(self) -> int:
        self._roll_day()
        return self._used_today

    @property
    def remaining_today(self) -> int:
        return max(0, self.daily_budget - self.used_today)

    def _roll_day(self) -> None:
        today = _utc_today(self._clock())
        if today != self._day:
            self._day, self._used_today = today, 0
            self._save_state()

    def _load_state(self) -> tuple[str, int]:
        today = _utc_today(self._clock())
        if self.state_path and self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("day") == today:
                return today, int(state.get("used", 0))
        return today, 0

    def _save_state(self) -> None:
        if self.state_path:
            payload = {"day": self._day, "used": self._used_today}
            atomic_write_text(self.state_path, json.dumps(payload))

    # Per minute window

    def _wait_for_window(self) -> None:
        while True:
            now = self._clock()
            while self._window and now - self._window[0] >= WINDOW_SECONDS:
                self._window.popleft()
            if len(self._window) < self.per_minute_limit:
                return
            self._sleep(WINDOW_SECONDS - (now - self._window[0]) + 0.05)

    # Public API

    def acquire(self) -> None:
        """Block until a request is allowed, then count it. Raises when the day is spent."""
        if self.remaining_today <= 0:
            raise DailyQuotaExhaustedError(
                f"Daily budget of {self.daily_budget} requests used "
                f"({self.daily_limit} limit, {self.safety_margin} margin). Resume after 00:00 UTC."
            )
        self._wait_for_window()
        self._window.append(self._clock())
        self._used_today += 1
        self._save_state()

    def observe_server_remaining(
        self, *, daily_remaining: int | None, minute_remaining: int | None = None
    ) -> None:
        """Reconcile with the quota the server reports. Only ever tightens our view."""
        if daily_remaining is not None:
            server_used = self.daily_limit - daily_remaining
            if server_used > self.used_today:
                self._used_today = server_used
                self._save_state()
        if minute_remaining is not None and minute_remaining <= 0:
            # Server says the minute is full: treat the window as full from now.
            now = self._clock()
            while len(self._window) < self.per_minute_limit:
                self._window.append(now)
