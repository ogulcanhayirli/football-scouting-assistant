"""One chat turn: Claude Sonnet 5.5 with the four tools, at most six tool calls.

* The request: ``claude-sonnet-5-5`` at effort ``medium`` (adaptive
  thinking, the model's default), the tool definitions with a cache
  breakpoint after the last one and the system prompt with its own, plus
  automatic caching of the growing conversation, so the second and later
  calls of a turn read the prefix from the cache. Server side fallback
  (``fallbacks: "default"``) lets the API rerun a request its safety
  classifiers decline on the model it recommends for that category.
* The loop: every ``tool_use`` block is answered with its result in one user
  message; after six tool calls the next request carries ``tool_choice:
  none``, so the model must answer with what it has. Tool calls past the six
  in one reply get an error result saying so. The clock does the same: after
  ``SOFT_SECONDS`` the next call must answer, and past ``HARD_SECONDS`` no
  call starts, so a turn ends well inside the Lambda's timeout and its cost
  is always recorded (an API error ends the turn the same way, keeping the
  cost of the calls already made).
* The answer: the text blocks of the final reply. A refusal by the API, an
  answer that repeats the rules (``scout.chat.prompt.leaks``), tool call
  markup written as text (a model stuck after its last call, C06 of the
  first evaluation) or a turn that runs out of budget gets a fixed reply
  instead.
* The money: each reply is priced from its usage fields (input, cache
  writes at 1.25 times, cache reads at 0.1 times, output), per model when a
  fallback model ran part of it.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from scout.chat import prompt, tools
from scout.chat.data import Tables

log = logging.getLogger(__name__)

MODEL = "claude-sonnet-5-5"
EFFORT = "medium"
MAX_TOOL_CALLS = 6
MAX_TOKENS = 6000  # per reply, thinking included
SOFT_SECONDS = 60  # after this the next call must answer
HARD_SECONDS = 90  # past this no call starts (each call: 40 s, one retry; Lambda: 180 s)
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# USD per million tokens: (input, output). Cache writes (five minutes) cost
# 1.25 times the input rate and cache reads 0.1 times.
PRICES = {
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
}
CACHE_WRITE, CACHE_READ = 1.25, 0.10
TOOL_MARKUP = re.compile(r"<(?:invoke|parameter|function_calls)\b")
USAGE = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")


def usage_of(u: Any) -> dict[str, int]:
    return {k: int(getattr(u, k, 0) or 0) for k in USAGE}


def price(usage: dict[str, int], model: str) -> float:
    # A model without a known rate is priced at the dearest one, never less.
    inp, out = PRICES.get(model, max(PRICES.values()))
    billed_in = (
        usage["input_tokens"]
        + CACHE_WRITE * usage["cache_creation_input_tokens"]
        + CACHE_READ * usage["cache_read_input_tokens"]
    )
    return (billed_in * inp + usage["output_tokens"] * out) / 1e6


def cost_usd(message: Any, model: str = MODEL) -> tuple[float, dict[str, int]]:
    """A reply's cost and tokens; per iteration when a fallback model ran."""
    u = getattr(message, "usage", None)
    iterations = getattr(u, "iterations", None) or []
    if iterations:
        total, tokens = 0.0, dict.fromkeys(USAGE, 0)
        for it in iterations:
            part = usage_of(it)
            total += price(part, getattr(it, "model", None) or model)
            for k in USAGE:
                tokens[k] += part[k]
        return total, tokens
    tokens = usage_of(u)
    return price(tokens, getattr(message, "model", None) or model), tokens


@dataclass
class Turn:
    answer: str
    outcome: str  # answered, refused_safety, leak_blocked, unfinished, budget, timeout, error
    tool_results: list[dict] = field(default_factory=list)
    calls: int = 0
    cost_usd: float = 0.0
    usage: dict[str, int] = field(default_factory=lambda: dict.fromkeys(USAGE, 0))
    seconds: float = 0.0


class Assistant:
    """The model, the tools and the prompt for one loaded set of tables."""

    def __init__(
        self, tables: Tables, client: Any, model: str = MODEL, clock=time.monotonic
    ) -> None:
        self.tables, self.client, self.model, self.clock = tables, client, model, clock
        self.definitions = tools.definitions(tables)
        self.definitions[-1] = {**self.definitions[-1], "cache_control": {"type": "ephemeral"}}
        self.system = prompt.system_blocks(tables)
        self.guarded = prompt.shingles(prompt.system_prompt(tables)[1])
        limits, topics = prompt.limits(tables), prompt.column_topics(tables)
        self.run_tool: dict[str, Callable[[dict], dict]] = {
            "search_players": lambda a: tools.search_players(tables, a),
            "player_card": lambda a: tools.player_card(tables, a),
            "similar_players": lambda a: tools.similar_players(tables, a),
            "how_to_read": lambda a: tools.how_to_read(tables, a, limits, topics),
        }

    def params(self, messages: list, final: bool) -> dict:
        p = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": self.system,
            "tools": self.definitions,
            "messages": messages,
            "output_config": {"effort": EFFORT},
            "cache_control": {"type": "ephemeral"},
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
        }
        if final:
            p["tool_choice"] = {"type": "none"}
        return p

    def call_tool(self, name: str, args: dict) -> tuple[dict, bool]:
        run = self.run_tool.get(name)
        if run is None:
            return {"error": f"no tool named {name}"}, True
        try:
            return run(args), False
        except Exception as exc:  # a bad argument must not end the turn
            log.warning("chat tool %s failed: %s", name, type(exc).__name__)
            return {
                "error": f"the tool could not run with these arguments ({type(exc).__name__})"
            }, True

    def answer(self, history: list[dict], message: str, room: Callable[[float], bool]) -> Turn:
        """``room(spent)``: whether the budget allows one more call after
        ``spent`` USD in this turn."""
        start = self.clock()
        turn = Turn(answer="", outcome="answered")
        messages = [*history, {"role": "user", "content": message}]
        for _ in range(MAX_TOOL_CALLS + 2):
            if not room(turn.cost_usd):
                turn.answer, turn.outcome = prompt.CAPPED["budget"], "budget"
                break
            elapsed = self.clock() - start
            if elapsed > HARD_SECONDS:
                turn.answer, turn.outcome = prompt.CAPPED["error"], "timeout"
                break
            final = turn.calls >= MAX_TOOL_CALLS or elapsed > SOFT_SECONDS
            try:
                reply = self.client.beta.messages.create(**self.params(messages, final))
            except Exception as exc:  # the API failed: end the turn, keep its cost
                log.warning("chat: the model call failed: %s", type(exc).__name__)
                turn.answer, turn.outcome = prompt.CAPPED["error"], "error"
                break
            cost, tokens = cost_usd(reply, self.model)
            turn.cost_usd += cost
            for k in USAGE:
                turn.usage[k] += tokens[k]
            if reply.stop_reason == "refusal":
                turn.answer, turn.outcome = prompt.CAPPED["safety"], "refused_safety"
                break
            uses = [b for b in reply.content if getattr(b, "type", None) == "tool_use"]
            if reply.stop_reason == "tool_use" and uses and not final:
                messages.append({"role": "assistant", "content": reply.content})
                results = []
                for block in uses:
                    if turn.calls >= MAX_TOOL_CALLS:
                        out, error = {"error": "tool call limit reached; answer now"}, True
                    else:
                        out, error = self.call_tool(block.name, dict(block.input))
                        turn.calls += 1
                        turn.tool_results.append(
                            {"tool": block.name, "input": dict(block.input), "result": out}
                        )
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(out, ensure_ascii=False),
                            **({"is_error": True} if error else {}),
                        }
                    )
                messages.append({"role": "user", "content": results})
                continue
            text = "\n\n".join(
                b.text.strip() for b in reply.content if getattr(b, "type", None) == "text"
            ).strip()
            if not text:
                turn.answer, turn.outcome = prompt.CAPPED["error"], "error"
            elif prompt.leaks(text, self.guarded):
                turn.answer, turn.outcome = prompt.REFUSAL, "leak_blocked"
            elif TOOL_MARKUP.search(text):
                turn.answer, turn.outcome = prompt.CAPPED["unfinished"], "unfinished"
            else:
                turn.answer = text
            break
        else:
            turn.answer, turn.outcome = prompt.CAPPED["error"], "error"
        turn.seconds = round(self.clock() - start, 2)
        return turn
