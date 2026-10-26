"""The Lambda behind the chat's Function URL (``infra/scout_infra/chat_stack.py``).

``POST`` a JSON body ``{"message": "...", "history": [{"role": "user" or
"assistant", "content": "..."}]}`` (the page sends the last few turns as
plain text); the reply is ``{"answer", "outcome", "tool_results", "usage",
"data_built"}``. ``tool_results`` are the tool calls the answer was built
from, which the page shows under it ("from the tables").

The published tables are read from the site bucket at cold start and kept
in memory; once an hour the handler asks S3 whether the page changed
(``HEAD index.html``) and reloads if it did. CORS is the Function URL's own
setting (the site's origin only), so the handler adds no CORS headers.

Environment: ``SITE_BUCKET``, ``STATE_BUCKET``, ``ANTHROPIC_API_KEY`` (set
at deploy time from a NoEcho stack parameter, never written to a file).
The logs carry the outcome, the tool names, tokens, cost and seconds, never
the message, the answer, the key or the prompt.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from typing import Any

from scout.chat import data, prompt
from scout.chat.agent import Assistant, Turn
from scout.chat.guard import Guard, S3Store, is_eval

log = logging.getLogger()
log.setLevel(logging.INFO)

HISTORY_TURNS = 6  # earlier messages kept, user and assistant together
HISTORY_CHARS = {"user": 500, "assistant": 2000}
RELOAD_SECONDS = 3600
EVAL_HEADER = "x-scout-eval"

_state: dict[str, Any] = {}


def _clients() -> tuple[Any, Any]:
    if "s3" not in _state:
        import anthropic
        import boto3

        _state["s3"] = boto3.client("s3")
        _state["claude"] = anthropic.Anthropic(max_retries=1, timeout=40.0)
    return _state["s3"], _state["claude"]


def assistant() -> Assistant:
    """The assistant over the current tables, reloaded when the page changed."""
    s3, claude = _clients()
    now = time.monotonic()
    current = _state.get("assistant")
    if current is not None and now - _state["checked"] < RELOAD_SECONDS:
        return current
    source = data.S3Source(s3, os.environ["SITE_BUCKET"])
    if current is not None and source.version() == current.tables.version:
        _state["checked"] = now
        return current
    tables = data.load(source)
    _state["assistant"], _state["checked"] = Assistant(tables, claude), now
    log.info("chat: tables of %s loaded", tables.built)
    return _state["assistant"]


def guard() -> Guard:
    if "guard" not in _state:
        s3, _ = _clients()
        _state["guard"] = Guard(S3Store(s3, os.environ["STATE_BUCKET"]))
    return _state["guard"]


def _reply(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json; charset=utf-8", "cache-control": "no-store"},
        "body": json.dumps(body, ensure_ascii=False),
    }


def _fixed(reason: str, status: int = 200) -> dict:
    return _reply(status, {"answer": prompt.CAPPED[reason], "outcome": reason, "tool_results": []})


def parse(event: dict) -> tuple[str, list[dict]]:
    """The message and the cleaned history; raises ValueError on a bad body."""
    body = event.get("body") or ""
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")
    payload = json.loads(body)
    message = payload.get("message")
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non empty string")
    history = []
    for item in (payload.get("history") or [])[-HISTORY_TURNS:]:
        role, content = item.get("role"), item.get("content")
        if role in HISTORY_CHARS and isinstance(content, str) and content.strip():
            history.append({"role": role, "content": content[: HISTORY_CHARS[role]]})
    while history and history[0]["role"] != "user":
        history.pop(0)  # the conversation must start with the user
    merged: list[dict] = []
    for item in history:  # no two turns of one role in a row
        if merged and merged[-1]["role"] == item["role"]:
            merged[-1] = item
        else:
            merged.append(item)
    if merged and merged[-1]["role"] == "user":
        merged.pop()  # the new message is the user's turn
    return message.strip(), merged


def handler(event: dict, context: Any = None) -> dict:
    http = (event.get("requestContext") or {}).get("http") or {}
    if http.get("method", "POST") != "POST":
        return _reply(405, {"error": "POST a JSON body"})
    try:
        message, history = parse(event)
    except (ValueError, AttributeError, TypeError):
        return _reply(400, {"error": 'send JSON: {"message": "...", "history": [...]}'})
    caps = guard().caps
    if len(message) > caps.max_chars:
        return _fixed("length")
    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    evaluation = is_eval(headers.get(EVAL_HEADER), os.environ.get("ANTHROPIC_API_KEY", ""))
    ip = http.get("sourceIp", "unknown")
    try:
        admitted = guard().admit(ip, evaluation)
    except Exception:
        log.exception("chat: state unavailable")
        return _fixed("error", 503)
    if not admitted.ok:
        log.info(json.dumps({"outcome": admitted.reason, "eval": evaluation}))
        return _fixed(admitted.reason)
    try:
        bot = assistant()
        turn = bot.answer(history, message, lambda spent: guard().room(admitted, spent))
    except Exception as exc:
        log.error("chat: turn failed: %s", type(exc).__name__)
        turn = Turn(answer=prompt.CAPPED["error"], outcome="error")
        bot = _state.get("assistant")
    try:
        guard().record(turn.cost_usd, turn.usage, evaluation)
    except Exception:
        log.exception("chat: cost not recorded")
    log.info(
        json.dumps(
            {
                "outcome": turn.outcome,
                "eval": evaluation,
                "tools": [r["tool"] for r in turn.tool_results],
                "usage": turn.usage,
                "usd": round(turn.cost_usd, 5),
                "seconds": turn.seconds,
                "chars": len(message),
            }
        )
    )
    return _reply(
        200,
        {
            "answer": turn.answer,
            "outcome": turn.outcome,
            "tool_results": turn.tool_results,
            "usage": {**turn.usage, "cost_usd": round(turn.cost_usd, 6), "seconds": turn.seconds},
            "data_built": bot.tables.built if bot is not None else None,
        },
    )
