"""Archetypes: players without data (prompt 18c).

An archetype is a player described from the historical record rather than
measured: one YAML file per player under ``archetypes/`` (``legends/`` for
the page, ``calibration/`` for the check), each of the 18 style measures as
a percentile against the regular players of his position group in his own
seasons, with a one line justification and the sources. Every file is a
draft by a Claude Code session awaiting the owner's review.

**Into the level free space.** A percentile becomes a standardised value
through our own regular player seasons from 2015 (one row per player and
season, the level kept): the value at that percentile of his position
group's distribution. His ability percentile is the stored version 2
weights over those values, placed in the same seasons' distribution of that
score. The level is then taken out with the curves our own players go
through and the result embedded with the stored model, so the archetype is
searched among today's players exactly as a measured season is. Every
result carries ``LABEL``.

**Calibration.** Six players we can measure were drafted the same way,
before their data was looked at: Modrić, Iniesta, Van Dijk and Kanté in our
own data, Messi at 25 and Henry at 26 in StatsBomb's (with the peer group
correction when ``scout greats`` applied it). Each drafted percentile is set
against the measured one on the same scale; a trait without spread in the
group (an outfield player's saves) or without data that season is not
compared. The verdict follows the owner's rule: within 15 points on most
traits, the method is calibrated to that degree; otherwise the archetypes
are illustrative only. A draft of 50 for every trait (no knowledge) is the
baseline.

``scout archetypes`` reloads every file, checks it, and writes
``data/processed/archetypes/`` and ``reports/archetypes.md``; ``scout site``
puts the archetypes on the page. Nothing here feeds a model of the lists.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scout.embeddings import STYLE_COLUMNS, style_inputs, unit, vector_rows
from scout.fsutil import atomic_write_text
from scout.scouting import POSITION_GROUPS

log = logging.getLogger(__name__)

ROOT = Path("archetypes")
KINDS = ("legends", "calibration")
OUT = "archetypes"  # processed/archetypes/
GROUPS = sorted(set(POSITION_GROUPS.values()))
SOURCES = ("api_football", "statsbomb")
LABEL = "Archetype from the record, not measured."
TOP = 10
WITHIN = 15.0  # the owner's rule: a draft within 15 percentile points of the measurement
FLAT_GAP = 1e-9  # a measure whose 10th and 90th percentiles are equal has no spread in the group
OUTFIELD_FLAT = ("saves_per90",)  # an outfield player makes none: no style in it


@dataclass
class Archetype:
    key: str  # the file name without .yaml
    kind: str  # "legends" or "calibration"
    name: str
    group: str
    era: str
    drafted: str
    reviewed: bool
    traits: dict[str, float]
    why: dict[str, str]
    sources: list[str]
    calibration: dict | None = None
    extra: dict = field(default_factory=dict)  # reviewed_by, reviewed_on and any other field


# The files


def check(data, path: Path, kind: str) -> list[str]:
    """Everything wrong with one parsed file (empty when it is complete)."""
    where = str(path)
    if not isinstance(data, dict):
        return [f"{where}: not a mapping of fields"]
    problems = []
    for name in ("name", "era", "drafted"):
        if not isinstance(data.get(name), str) or not data[name].strip():
            problems.append(f"{where}: `{name}` is missing or empty")
    if data.get("position_group") not in GROUPS:
        problems.append(f"{where}: `position_group` must be one of {', '.join(GROUPS)}")
    if data.get("measured") is not False:
        problems.append(
            f"{where}: `measured` must be false (an archetype is drawn from the record)"
        )
    if not isinstance(data.get("reviewed"), bool):
        problems.append(f"{where}: `reviewed` must be true or false")
    traits = data.get("traits")
    if not isinstance(traits, dict):
        problems.append(f"{where}: `traits` is missing")
        traits = {}
    for trait in STYLE_COLUMNS:
        entry = traits.get(trait)
        if not isinstance(entry, dict):
            problems.append(f"{where}: trait `{trait}` is missing")
            continue
        pct = entry.get("percentile")
        if isinstance(pct, bool) or not isinstance(pct, (int, float)) or not 0 <= pct <= 100:
            problems.append(f"{where}: trait `{trait}` needs a percentile from 0 to 100")
        if not isinstance(entry.get("why"), str) or not entry["why"].strip():
            problems.append(f"{where}: trait `{trait}` needs a one line `why`")
    for trait in traits:
        if trait not in STYLE_COLUMNS:
            problems.append(f"{where}: unknown trait `{trait}`")
    sources = data.get("sources")
    if (
        not isinstance(sources, list)
        or not sources
        or not all(isinstance(s, str) and s.strip() for s in sources)
    ):
        problems.append(f"{where}: `sources` needs at least one source")
    cal = data.get("calibration")
    if kind == "calibration":
        if (
            not isinstance(cal, dict)
            or cal.get("source") not in SOURCES
            or not isinstance(cal.get("player_id"), int)
            or not isinstance(cal.get("season"), int)
        ):
            sources = " or ".join(SOURCES)
            problems.append(
                f"{where}: `calibration` needs source ({sources}), player_id and season"
            )
    elif cal is not None:
        problems.append(f"{where}: only files under calibration/ carry `calibration`")
    return problems


def load(root: Path = ROOT) -> list[Archetype]:
    """Every archetype file under ``root``/legends and ``root``/calibration,
    checked; raises ValueError naming every problem found."""
    root = Path(root)
    found, problems = [], []
    for kind in KINDS:
        for path in sorted((root / kind).glob("*.yaml")):
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
            except yaml.YAMLError as exc:
                problems.append(f"{path}: not valid YAML ({exc})")
                continue
            issues = check(data, path, kind)
            if issues:
                problems += issues
                continue
            known = {
                "name",
                "position_group",
                "era",
                "measured",
                "drafted",
                "reviewed",
                "traits",
                "sources",
                "calibration",
            }
            found.append(
                Archetype(
                    key=path.stem,
                    kind=kind,
                    name=data["name"],
                    group=data["position_group"],
                    era=data["era"],
                    drafted=data["drafted"],
                    reviewed=data["reviewed"],
                    traits={t: float(data["traits"][t]["percentile"]) for t in STYLE_COLUMNS},
                    why={t: data["traits"][t]["why"].strip() for t in STYLE_COLUMNS},
                    sources=list(data["sources"]),
                    calibration=data.get("calibration"),
                    extra={k: v for k, v in data.items() if k not in known},
                )
            )
    if problems:
        raise ValueError("archetype files need fixing:\n" + "\n".join(problems))
    if not found:
        raise FileNotFoundError(f"no archetype files under {root}/legends or {root}/calibration")
    return found


# The scale


@dataclass
class TraitScale:
    """Per position group: the sorted standardised values (level kept) of
    every measure over our regular player seasons from 2015, one row per
    player and season, and the sorted ability score (stored version 2
    weights over them). A percentile becomes a value and a value a
    percentile on the same scale."""

    values: dict[str, dict[str, np.ndarray]]
    composite: dict[str, np.ndarray]

    def value(self, group: str, measure: str, pct: float) -> float:
        return float(np.quantile(self.values[group][measure], pct / 100))

    def percentile(self, group: str, measure: str, z: float) -> float:
        """Mid rank: half of the tied values count below."""
        ref = self.values[group][measure]
        below = np.searchsorted(ref, z, side="left")
        upto = np.searchsorted(ref, z, side="right")
        return float(100 * (below + upto) / 2 / len(ref))

    def flat(self, group: str, measure: str) -> bool:
        """No spread to compare: an outfield player's saves (each season's
        zero standardises a little differently, which is not style), or a
        measure whose 10th and 90th percentiles are equal."""
        if group != "goalkeeper" and measure in OUTFIELD_FLAT:
            return True
        ref = self.values[group][measure]
        return bool(np.quantile(ref, 0.9) - np.quantile(ref, 0.1) <= FLAT_GAP)

    def ability(self, group: str, score: float) -> float:
        ref = self.composite[group]
        return float(100 * np.searchsorted(ref, score, side="right") / len(ref))


def build_scale(kept, weights: pd.DataFrame) -> TraitScale:
    """The scale from the standardised measures with the level kept
    (``style_vectors(..., keep_level=True)``) and the stored ability weights."""
    from scout.greats import composite_score

    main = kept.meta["main"].to_numpy(dtype=bool)
    meta = kept.meta[main].reset_index(drop=True)
    z = pd.DataFrame(kept.style[main], columns=kept.columns)
    score = composite_score(z, meta, weights)
    values, composite = {}, {}
    for g in sorted(meta["group"].dropna().unique()):
        at = (meta["group"] == g).to_numpy()
        values[g] = {c: np.sort(z.loc[at, c].to_numpy(dtype=float)) for c in kept.columns}
        composite[g] = np.sort(score[at].dropna().to_numpy(dtype=float))
    return TraitScale(values, composite)


# Into the level free space


def archetype_vectors(archetypes: list[Archetype], scale: TraitScale, own) -> pd.DataFrame:
    """One row per archetype: its standardised values (``z_<measure>``), its
    ability percentile, the level free style (``style_<measure>``) and the
    embedding (``emb_*``), through the stored level curves and model."""
    from scout.greats import composite_score

    columns = own.stored.columns
    z = pd.DataFrame(
        [[scale.value(a.group, c, a.traits[c]) for c in columns] for a in archetypes],
        columns=columns,
    )
    meta = pd.DataFrame(
        {
            "key": [a.key for a in archetypes],
            "kind": [a.kind for a in archetypes],
            "name": [a.name for a in archetypes],
            "group": [a.group for a in archetypes],
            "sub_position": [_position(a.group) for a in archetypes],
        }
    )
    score = composite_score(z, meta, own.weights)
    meta["ability_pct"] = [scale.ability(g, s) for g, s in zip(meta["group"], score, strict=True)]
    meta["ability_pct_1s"] = meta["ability_pct"]
    style = own.curves.apply(z.to_numpy(), meta[["ability_pct", "ability_pct_1s"]], meta["group"])
    emb = own.stored.vectors(style)
    return pd.concat(
        [
            meta.drop(columns="sub_position"),
            z.add_prefix("z_"),
            pd.DataFrame(style, columns=[f"style_{c}" for c in columns]),
            pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(emb.shape[1])]),
        ],
        axis=1,
    )


def _position(group: str) -> str:
    """A Transfermarkt position of the group (position_group reads it back)."""
    return next(p for p, g in POSITION_GROUPS.items() if g == group)


def nearest(
    row: pd.Series, today: pd.DataFrame, k: int = TOP, only: set | None = None
) -> pd.DataFrame:
    """Today's players of the archetype's position group nearest to it; with
    ``only``, among those player ids (the elite pool). Every row carries the
    label."""
    emb = [c for c in today.columns if c.startswith("emb_")]
    pool = today[today["group"] == row["group"]]
    if only is not None:
        pool = pool[pool["source_player_id"].astype(int).isin(only)]
    query = unit(row[emb].to_numpy(dtype=float)[None, :])[0]
    sims = unit(pool[emb].to_numpy(dtype=float)) @ query
    order = np.argsort(-sims, kind="stable")[:k]
    found = pool.iloc[order].copy()
    found.insert(0, "rank", np.arange(1, len(order) + 1))
    found["similarity"] = sims[order]
    found["label"] = LABEL
    return found.reset_index(drop=True)


def neighbours(vectors: pd.DataFrame, today: pd.DataFrame, k: int = TOP) -> pd.DataFrame:
    """Today's ``k`` nearest players of every archetype."""
    out = []
    for _, row in vectors.iterrows():
        found = nearest(row, today, k)
        out.append(
            pd.DataFrame(
                {
                    "archetype": row["key"],
                    "name": row["name"],
                    "group": row["group"],
                    "rank": found["rank"],
                    "player_id": found["source_player_id"].astype(int),
                    "player": found["player_name"],
                    "age_mid": found["age_mid"].astype(float),
                    "club": found["team_name"],
                    "league_key": found["league_key"],
                    "ability_pct": found["ability_pct"].astype(float),
                    "similarity": found["similarity"],
                    "label": LABEL,
                }
            )
        )
    return pd.concat(out, ignore_index=True)


# Calibration


def measured_rows(own, sb_vectors: pd.DataFrame | None) -> dict:
    """What the measurement reads: our main rows with their standardised
    values (level kept) and which inputs the season has, and the StatsBomb
    rows with theirs."""
    kept = own.kept
    rows = vector_rows(own.features, own.ability)
    inputs = style_inputs(own.features)[kept.columns][rows].reset_index(drop=True)
    return {"kept": kept, "present": inputs.notna(), "statsbomb": sb_vectors}


def measured(arch: Archetype, rows: dict) -> tuple[str, dict[str, float], dict[str, bool]] | None:
    """(group, standardised values, which inputs exist) of the calibration
    season, or None when it is not found."""
    cal = arch.calibration
    if cal["source"] == "api_football":
        kept = rows["kept"]
        meta = kept.meta
        at = np.flatnonzero(
            (meta["source_player_id"].astype(int) == cal["player_id"]).to_numpy()
            & (meta["season"].astype(int) == cal["season"]).to_numpy()
            & meta["main"].to_numpy(dtype=bool)
        )
        if not len(at):
            return None
        i = int(at[0])
        z = dict(zip(kept.columns, kept.style[i].astype(float), strict=True))
        present = rows["present"].iloc[i].to_dict()
        return str(meta.at[i, "group"]), z, present
    sb = rows["statsbomb"]
    if sb is None:
        return None
    hit = sb[(sb["source_player_id"] == cal["player_id"]) & (sb["season"] == cal["season"])]
    if hit.empty:
        return None
    r = hit.sort_values("minutes", ascending=False).iloc[0]
    z = {c: float(r[f"z_{c}"]) for c in STYLE_COLUMNS if f"z_{c}" in r}
    return str(r["group"]), z, {c: f"z_{c}" in r for c in STYLE_COLUMNS}


def calibration_table(archetypes: list[Archetype], scale: TraitScale, rows: dict) -> pd.DataFrame:
    """Per calibration archetype and trait: drafted and measured percentile,
    the gap, the baseline's gap (50 for every trait), and whether the trait
    is compared (spread in the group and data that season)."""
    out = []
    for a in archetypes:
        if a.kind != "calibration":
            continue
        found = measured(a, rows)
        if found is None:
            log.warning("archetypes: no measured season for %s (%s)", a.name, a.calibration)
            continue
        group, z, present = found
        for c in STYLE_COLUMNS:
            pct = scale.percentile(group, c, z[c]) if c in z else np.nan
            compared = bool(present.get(c, False)) and not scale.flat(group, c) and pd.notna(pct)
            out.append(
                {
                    "key": a.key,
                    "name": a.name,
                    "source": a.calibration["source"],
                    "season": a.calibration["season"],
                    "group": group,
                    "drafted_group": a.group,
                    "trait": c,
                    "drafted": a.traits[c],
                    "measured": pct,
                    "gap": abs(a.traits[c] - pct) if pd.notna(pct) else np.nan,
                    "baseline_gap": abs(50.0 - pct) if pd.notna(pct) else np.nan,
                    "compared": compared,
                }
            )
    return pd.DataFrame(out)


def own_season_ranks(
    archetypes: list[Archetype], vectors: pd.DataFrame, own_vectors: pd.DataFrame
) -> pd.DataFrame:
    """Each calibration archetype in our own data searched among every main
    row of his measured season and position group: the rank of his own
    season (1 is the nearest), as the identity checks rank."""
    emb = [c for c in vectors.columns if c.startswith("emb_")]
    out = []
    for a in archetypes:
        cal = a.calibration
        if a.kind != "calibration" or cal["source"] != "api_football":
            continue
        row = vectors[vectors["key"] == a.key].iloc[0]
        main = own_vectors[own_vectors["main"].astype(bool)]
        mine = main[
            (main["source_player_id"].astype(int) == cal["player_id"])
            & (main["season"].astype(int) == cal["season"])
        ]
        if mine.empty:
            continue
        group = mine["group"].iloc[0]
        pool = main[(main["season"].astype(int) == cal["season"]) & (main["group"] == group)]
        sims = (
            unit(pool[emb].to_numpy(dtype=float)) @ unit(row[emb].to_numpy(dtype=float)[None, :])[0]
        )
        true = sims[(pool["source_player_id"].astype(int) == cal["player_id"]).to_numpy()][0]
        out.append(
            {
                "name": a.name,
                "season": cal["season"],
                "group": group,
                "rank": int(1 + (sims > true).sum()),
                "pool": len(pool),
                "similarity": float(true),
            }
        )
    return pd.DataFrame(out)


def calibration_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Per player, the players in our own data, and all (the verdict's row,
    last), on compared traits: traits, mean absolute gap and share within
    ``WITHIN`` points, for the drafts and the baseline."""
    t = table[table["compared"]]
    rows = []
    for (_, name), g in t.groupby(["key", "name"], sort=False):
        rows.append(_summary_row(name, g))
    own = t[t["source"] == "api_football"]
    if len(own) and len(own) < len(t):
        rows.append(_summary_row(f"the {own['key'].nunique()} in our own data", own))
    rows.append(_summary_row(f"all {t['key'].nunique()}", t))
    return pd.DataFrame(rows)


def _summary_row(name: str, g: pd.DataFrame) -> dict:
    return {
        "name": name,
        "traits": len(g),
        "mean_gap": float(g["gap"].mean()) if len(g) else np.nan,
        "within": float((g["gap"] <= WITHIN).mean()) if len(g) else np.nan,
        "baseline_mean_gap": float(g["baseline_gap"].mean()) if len(g) else np.nan,
        "baseline_within": float((g["baseline_gap"] <= WITHIN).mean()) if len(g) else np.nan,
    }


def verdict(summary: pd.DataFrame) -> dict:
    """The owner's rule on the pooled traits (the last row): within 15 points
    on most (more than half), calibrated to that degree; else illustrative
    only."""
    if summary.empty or summary.iloc[-1]["traits"] == 0:
        return {"calibrated": False, "text": "Not checked: no calibration season was found."}
    all_ = summary.iloc[-1]
    groups = summary["name"].str.startswith(("all ", "the "))
    players = int((~groups).sum())
    calibrated = bool(all_["within"] > 0.5)
    numbers = (
        f"Drafted the same way for {players} players we can measure, before looking at their "
        f"data, the percentiles landed within {WITHIN:.0f} points of the measured ones on "
        f"{100 * all_['within']:.0f}% of {int(all_['traits'])} traits (mean gap "
        f"{all_['mean_gap']:.0f} points), against {100 * all_['baseline_within']:.0f}% "
        f"(mean gap {all_['baseline_mean_gap']:.0f}) for a draft of 50 on every trait."
    )
    own = summary[summary["name"].str.startswith("the ")]
    if len(own):
        o = own.iloc[0]
        numbers += (
            f" On {o['name']} ({int(o['traits'])} traits, the firmest "
            f"measurements): {100 * o['within']:.0f}% against {100 * o['baseline_within']:.0f}%."
        )
    text = (
        f"{numbers} The method is calibrated to that degree: read each trait as within about "
        f"{WITHIN:.0f} points most of the time, and wrong by more on the rest."
        if calibrated
        else f"{numbers} That is not most traits, so the archetypes are illustrative only: a "
        "sketch of a style from the record, not a measurement."
    )
    return {"calibrated": calibrated, "text": text}


# The command


def run_archetypes(settings, root: Path = ROOT) -> Path:
    """``scout archetypes``: reload and check every file, the vectors, today's
    nearest, the calibration check, ``data/processed/archetypes/`` and
    ``reports/archetypes.md``."""
    from scout.greats import OUT as SB_OUT
    from scout.greats import own_data, today_pool
    from scout.lists import league_names

    archetypes = load(root)
    processed = settings.paths.processed_dir
    own = own_data(settings)
    scale = build_scale(own.kept, own.weights)
    vectors = archetype_vectors(archetypes, scale, own)
    calendar = [lg.key for lg in settings.leagues if lg.calendar_year]
    today = today_pool(own, calendar)
    near = neighbours(vectors, today)
    sb_path = processed / SB_OUT / "vectors.parquet"
    sb = pd.read_parquet(sb_path) if sb_path.is_file() else None
    if sb is not None and not any(c.startswith("z_") for c in sb.columns):
        sb = None  # built before prompt 18c: run `scout greats`
    if sb is None:
        log.warning("archetypes: no StatsBomb values (run `scout greats`); Messi and Henry skipped")
    greats_info = processed / "greats" / "greats.json"
    corrected = (
        json.loads(greats_info.read_text()).get("correction", {}).get("applied", False)
        if greats_info.is_file()
        else False
    )
    table = calibration_table(archetypes, scale, measured_rows(own, sb))
    summary = calibration_summary(table)
    ranks = own_season_ranks(archetypes, vectors, own.vectors)
    decision = verdict(summary)

    out = processed / OUT
    out.mkdir(parents=True, exist_ok=True)
    emb = [c for c in vectors.columns if c.startswith("emb_")]
    vectors.assign(**{c: vectors[c].astype("float32") for c in emb}).to_parquet(
        out / "archetypes.parquet", index=False
    )
    near.to_parquet(out / "neighbours.parquet", index=False)
    table.to_parquet(out / "calibration.parquet", index=False)
    info = {
        "label": LABEL,
        "calibration": {**decision, "within": WITHIN, "summary": _records(summary)},
        "statsbomb_corrected": bool(corrected),
        "archetypes": [
            {
                "key": a.key,
                "kind": a.kind,
                "name": a.name,
                "group": a.group,
                "era": a.era,
                "drafted": a.drafted,
                "reviewed": a.reviewed,
                **a.extra,
                "ability_pct": float(vectors.loc[i, "ability_pct"]),
                "traits": [[t, a.traits[t], a.why[t]] for t in STYLE_COLUMNS],
                "sources": a.sources,
                "calibration": a.calibration,
            }
            for i, a in enumerate(archetypes)
        ],
        "created": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"),
    }
    atomic_write_text(
        out / "archetypes.json", json.dumps(info, indent=1, ensure_ascii=False, default=str) + "\n"
    )
    path = write_report(
        settings.paths.reports_dir,
        archetypes=archetypes,
        vectors=vectors,
        near=near,
        table=table,
        summary=summary,
        ranks=ranks,
        decision=decision,
        corrected=corrected,
        names=league_names(settings),
    )
    log.info(
        "archetypes: %d files (%d legends), calibration within %.0f on %.0f%% of traits; %s",
        len(archetypes),
        sum(a.kind == "legends" for a in archetypes),
        WITHIN,
        100 * summary.iloc[-1]["within"] if len(summary) else float("nan"),
        path,
    )
    return path


def _records(frame: pd.DataFrame) -> list[dict]:
    return [
        {k: (v.item() if isinstance(v, np.generic) else v) for k, v in row.items()}
        for row in frame.to_dict(orient="records")
    ]


# The report

TRAIT_NAMES = {
    "goals_per90": "goals",
    "assists_per90": "assists",
    "shots_total_per90": "shots",
    "passes_key_per90": "key passes",
    "passes_total_per90": "passes",
    "dribbles_success_per90": "dribbles",
    "dribble_success_rate": "dribble success",
    "duels_won_per90": "duels won",
    "duel_win_rate": "duel win rate",
    "tackles_total_per90": "tackles",
    "interceptions_per90": "interceptions",
    "blocks_per90": "blocks",
    "fouls_drawn_per90": "fouls drawn",
    "saves_per90": "saves",
    "share_shots": "share of shots",
    "share_dribbles": "share of dribbles",
    "share_passes": "share of passes",
    "share_duels": "share of duels",
}


def _season(s: int) -> str:
    return f"{s}/{str(s + 1)[-2:]}"


def write_report(reports_dir: Path, **k) -> Path:
    archetypes, vectors, near = k["archetypes"], k["vectors"], k["near"]
    table, summary, decision, names = k["table"], k["summary"], k["decision"], k["names"]
    lines = [
        "# Archetypes (prompt 18c)",
        "",
        "Written by `scout archetypes` (`src/scout/archetypes.py`) from the files under "
        "`archetypes/`. An archetype is a player described from the historical record, each "
        "of the 18 style measures as a percentile against the regulars of his position group "
        "in his own seasons, with a one line reason and the sources. **Every file was drafted "
        "by a Claude Code session on 25 October 2026 and awaits the owner's review**; the "
        "session watched no matches. The percentiles become standardised values on our own "
        "regular player seasons from 2015, the level is taken out with the curves our own "
        "players go through, and the stored model embeds the result: the archetype is "
        "searched among today's players as a measured season is.",
        "",
        f"_{LABEL}_",
        "",
        "## Calibration",
        "",
        "Six players we can measure were drafted the same way, before their data was looked "
        "at (the drafts were committed on their own first): Modrić, Iniesta, Van Dijk and "
        "Kanté in our own data, Messi at 25 and Henry at 26 in StatsBomb's"
        + (" (peer group corrected, `scout greats`)" if k["corrected"] else "")
        + ". Each drafted percentile is set against the measured one on the same scale "
        "(his standardised value placed among our regular player seasons of his position "
        "group from 2015). A trait without spread in the group (an outfield player's saves) "
        "or without data that season (API Football's La Liga has no tackles or blocks "
        f"before 2017/18) is not compared. Within {WITHIN:.0f} is the owner's rule; the "
        "baseline drafts 50 on every trait.",
        "",
        "| Player | Season | Traits compared | Mean gap | Within 15 | Baseline mean gap "
        "| Baseline within 15 |",
        "|---|---|---|---|---|---|---|",
    ]
    seasons = (
        table.drop_duplicates("name").set_index("name")[["source", "season"]]
        if len(table)
        else pd.DataFrame()
    )
    for r in summary.itertuples(index=False):
        if r.name in seasons.index:
            s = seasons.loc[r.name]
            src = "StatsBomb" if s["source"] == "statsbomb" else "API Football"
            when = f"{_season(int(s['season']))}, {src}"
        else:
            when = ""
        name = f"**{r.name}**" if r.name.startswith(("all ", "the ")) else r.name
        lines.append(
            f"| {name} | {when} | {r.traits} | {r.mean_gap:.1f} | {100 * r.within:.0f}% | "
            f"{r.baseline_mean_gap:.1f} | {100 * r.baseline_within:.0f}% |"
        )
    lines += ["", f"**Verdict.** {decision['text']}", ""]
    ranks = k["ranks"]
    if len(ranks):
        lines += [
            "Searched with his drafted archetype among every regular of his position group "
            "that season (the identity check's question: is the real season among the five "
            "nearest?):",
            "",
            "| Player | Season | His own season's rank | Pool | Similarity | Random top 5 |",
            "|---|---|---|---|---|---|",
        ]
        for r in ranks.itertuples(index=False):
            lines.append(
                f"| {r.name} | {_season(r.season)} | {r.rank} | {r.pool:,} | "
                f"{r.similarity:.2f} | {100 * min(5, r.pool) / r.pool:.1f}% |"
            )
        lines += [
            "",
            "The StatsBomb seasons (Messi, Henry) are not searched: a season held for one "
            "club has no pool of other regulars.",
            "",
        ]
    if len(table):
        players = list(dict.fromkeys(table["name"]))
        lines += [
            "Each trait, drafted against measured (n.c.: not compared):",
            "",
            "| Trait | " + " | ".join(players) + " |",
            "|---|" + "---|" * len(players),
        ]
        cells = table.set_index(["trait", "name"])
        for t in STYLE_COLUMNS:
            row = []
            for p in players:
                r = cells.loc[(t, p)]
                if r["compared"]:
                    far = " ●" if r["gap"] > WITHIN else ""
                    row.append(f"{r['drafted']:.0f} / {r['measured']:.0f}{far}")
                else:
                    row.append(f"{r['drafted']:.0f} / n.c.")
            lines.append(f"| {TRAIT_NAMES.get(t, t)} | " + " | ".join(row) + " |")
        lines += ["", f"● the draft is more than {WITHIN:.0f} points away.", ""]
    lines += [
        "## The archetypes and today's nearest",
        "",
        "Today's players (the newest season) of the archetype's position group, any age, "
        "nearest by the learned embedding. Ability is the percentile of their own season; "
        "similar means plays the same way, not as good.",
        "",
    ]
    for a in archetypes:
        v = vectors[vectors["key"] == a.key].iloc[0]
        top = near[near["archetype"] == a.key].head(5)
        kind = " (calibration draft)" if a.kind == "calibration" else ""
        lines += [
            f"### {a.name}{kind}",
            "",
            f"{a.era}; {a.group.replace('_', ' ')}; ability {v['ability_pct']:.0f} percentile "
            "from the drafted traits. Highest drafted: "
            + ", ".join(
                f"{TRAIT_NAMES[t]} {a.traits[t]:.0f}"
                for t in sorted(STYLE_COLUMNS, key=lambda t: -a.traits[t])[:4]
            )
            + "; lowest: "
            + ", ".join(
                f"{TRAIT_NAMES[t]} {a.traits[t]:.0f}"
                for t in sorted(STYLE_COLUMNS, key=lambda t: a.traits[t])[:3]
            )
            + ".",
            "",
            "| # | Player | Age | Club | League | Ability | Similarity |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in top.itertuples(index=False):
            lines.append(
                f"| {r.rank} | {r.player} | {int(np.floor(r.age_mid))} | {r.club} | "
                f"{names.get(r.league_key, r.league_key)} | {r.ability_pct:.0f} | "
                f"{r.similarity:.2f} |"
            )
        lines += ["", f"_{LABEL}_", ""]
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "archetypes.md"
    atomic_write_text(path, "\n".join(lines))
    return path
