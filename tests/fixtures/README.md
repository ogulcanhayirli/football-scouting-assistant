# Test fixtures

Everything here is **synthetic**: real response shapes, made up players, clubs
and numbers. Player ids start at 900000 so they can't collide with real ids.
The shapes follow the API Football v3 `/players` and `/leagues` responses and
the transfermarkt-datasets CSV columns.

Error fixtures approximate the wording API Football uses for plan, quota, per minute
and token errors. All of them come back as HTTP 200, with the problem reported
in the `errors` field. They will be replaced with real cached responses after the
feasibility run.
