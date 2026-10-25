"""Archetypes from the record: the files, the scale, the vectors, the calibration (prompt 18c)."""

import base64
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scout import archetypes as ar
from scout.embeddings import STYLE_COLUMNS, StoredEmbedding
from scout.greats import LevelCurves, OwnData
from scout.site import archetypes_data, similar_data

ROOT = Path(__file__).resolve().parents[2] / "archetypes"
LEGENDS = {
    "Johan Cruyff",
    "Ruud Gullit",
    "Rio Ferdinand",
    "Paolo Maldini",
    "Zinedine Zidane",
    "Ronaldo Nazário",
    "Franco Baresi",
    "Andrea Pirlo",
}
CALIBRATION = {
    "Luka Modrić",
    "Andrés Iniesta",
    "Virgil van Dijk",
    "Lionel Messi at 25",
    "Thierry Henry at 26",
    "N'Golo Kanté",
}


def test_every_archetype_file_has_every_trait_and_a_source():
    paths = sorted(ROOT.glob("*/*.yaml"))
    assert len(paths) == len(LEGENDS) + len(CALIBRATION)
    for path in paths:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert set(data["traits"]) == set(STYLE_COLUMNS), path
        for trait, entry in data["traits"].items():
            assert 0 <= entry["percentile"] <= 100, (path, trait)
            assert entry["why"].strip(), (path, trait)
        assert data["sources"], path
        assert all(s.strip() for s in data["sources"]), path
        assert data["measured"] is False, path
        assert "awaits the owner's review" in data["drafted"] or data["reviewed"], path
    found = ar.load(ROOT)  # the loader's own check passes on every file
    assert {a.name for a in found if a.kind == "legends"} == LEGENDS
    assert {a.name for a in found if a.kind == "calibration"} == CALIBRATION
    assert all(a.calibration["source"] in ar.SOURCES for a in found if a.kind == "calibration")


def _write(root, kind, name, data):
    (root / kind).mkdir(parents=True, exist_ok=True)
    (root / kind / f"{name}.yaml").write_text(yaml.safe_dump(data, allow_unicode=True))


def _file(**over):
    data = {
        "name": "Test",
        "position_group": "winger",
        "era": "1990s",
        "measured": False,
        "drafted": "Drafted on a test day; awaits the owner's review.",
        "reviewed": False,
        "traits": {t: {"percentile": 50, "why": "Typical."} for t in STYLE_COLUMNS},
        "sources": ["A book."],
    }
    data.update(over)
    return data


def test_a_file_missing_a_trait_or_a_source_is_refused(tmp_path):
    bad = _file(sources=[])
    del bad["traits"]["share_duels"]
    bad["traits"]["goals_per90"] = {"percentile": 120, "why": ""}
    _write(tmp_path, "legends", "bad", bad)
    _write(tmp_path, "calibration", "nocal", _file())
    _write(tmp_path, "legends", "measured", _file(measured=True))
    with pytest.raises(ValueError, match="archetype files need fixing") as err:
        ar.load(tmp_path)
    text = str(err.value)
    assert "trait `share_duels` is missing" in text
    assert "trait `goals_per90` needs a percentile from 0 to 100" in text
    assert "trait `goals_per90` needs a one line `why`" in text
    assert "`sources` needs at least one source" in text
    assert "`calibration` needs source" in text
    assert "`measured` must be false" in text


def test_an_edited_file_is_read_back_as_written(tmp_path):
    data = _file(name="Edited", reviewed=True, reviewed_by="Owner", reviewed_on="2026-11-01")
    data["traits"]["passes_total_per90"] = {"percentile": 99, "why": "Watched three matches."}
    _write(tmp_path, "legends", "edited", data)
    (found,) = ar.load(tmp_path)
    assert found.traits["passes_total_per90"] == 99
    assert found.why["passes_total_per90"] == "Watched three matches."
    assert found.reviewed
    assert found.extra["reviewed_by"] == "Owner"


def _scale():
    rng = np.random.default_rng(4)
    values = {
        g: {c: np.sort(rng.normal(0, 1, 2000)) for c in STYLE_COLUMNS}
        for g in ("winger", "goalkeeper")
    }
    values["winger"]["saves_per90"] = np.sort(-0.1 + rng.normal(0, 0.01, 2000))
    composite = {g: np.sort(rng.normal(0, 0.6, 2000)) for g in values}
    return ar.TraitScale(values, composite)


def test_the_scale_turns_a_percentile_into_a_value_and_back():
    scale = _scale()
    z = scale.value("winger", "goals_per90", 90)
    assert z == pytest.approx(1.28, abs=0.1)
    assert scale.percentile("winger", "goals_per90", z) == pytest.approx(90, abs=0.5)
    assert scale.flat("winger", "saves_per90")  # an outfield player's saves: no style
    assert not scale.flat("goalkeeper", "saves_per90")
    assert not scale.flat("winger", "goals_per90")
    ties = ar.TraitScale({"winger": {"m": np.array([0.0, 0.0, 0.0, 1.0])}}, {})
    assert ties.percentile("winger", "m", 0.0) == 37.5  # mid rank of the tied values


def _own(columns):
    k = len(columns)
    coef = {"winger": np.zeros((7, k))}
    coef["winger"][1] = 1.0  # the level adds the ability percentile to every measure
    spread = np.zeros((7, k))
    spread[0] = 1.0  # the same spread at every level
    curves = LevelCurves(coef, {"winger": spread}, {"winger": np.ones(k)})
    weights = pd.DataFrame({"position": ["winger"], "measure": ["goals_per90"], "weight": [1.0]})
    stored = StoredEmbedding(list(columns), {}, "baseline", None, {})
    return OwnData(stored, weights, {}, curves, pd.DataFrame(), pd.DataFrame())


def _archetype(key, kind, pct, group="winger", calibration=None):
    return ar.Archetype(
        key=key,
        kind=kind,
        name=key.title(),
        group=group,
        era="1990s",
        drafted="Drafted; awaits the owner's review.",
        reviewed=False,
        traits={t: pct.get(t, 50.0) for t in STYLE_COLUMNS},
        why={t: "Why." for t in STYLE_COLUMNS},
        sources=["A book."],
        calibration=calibration,
    )


def test_an_archetype_goes_through_the_level_free_space_and_finds_todays_players():
    scale = _scale()
    own = _own(STYLE_COLUMNS)
    archs = [
        _archetype("scorer", "legends", {"goals_per90": 99, "shots_total_per90": 95}),
        _archetype("passer", "legends", {"passes_total_per90": 99, "goals_per90": 10}),
    ]
    v = ar.archetype_vectors(archs, scale, own)
    assert v.loc[0, "ability_pct"] > 95 > v.loc[1, "ability_pct"]  # ability from goals only
    assert v.loc[0, "z_goals_per90"] == pytest.approx(scale.value("winger", "goals_per90", 99))
    # the level (ability / 100 on every measure here) is taken out before embedding
    expected = v.loc[0, "z_shots_total_per90"] - v.loc[0, "ability_pct"] / 100
    assert v.loc[0, "style_shots_total_per90"] == pytest.approx(expected)
    emb = [c for c in v.columns if c.startswith("emb_")]
    assert np.linalg.norm(v.loc[0, emb].astype(float)) == pytest.approx(1.0)
    today = pd.DataFrame(
        {
            "source_player_id": [1, 2, 3],
            "player_name": ["Scorer today", "Passer today", "Keeper"],
            "group": ["winger", "winger", "goalkeeper"],
        }
    )
    today = pd.concat(
        [today, pd.DataFrame(v[emb].to_numpy()[[0, 1, 0]], columns=emb)], axis=1
    ).assign(age_mid=24.0, team_name="Club", league_key="x", ability_pct=50.0)
    found = ar.nearest(v.iloc[0], today)
    assert found["player_name"].tolist() == ["Scorer today", "Passer today"]  # wingers only
    assert (found["label"] == ar.LABEL).all()
    assert ar.LABEL == "Archetype from the record, not measured."
    near = ar.neighbours(v, today)
    assert set(near["label"]) == {ar.LABEL}
    assert near.groupby("archetype")["rank"].max().to_dict() == {"passer": 2, "scorer": 2}


def _table(gaps_in, gaps_out):
    rows = []
    for i, g in enumerate(gaps_in + gaps_out):
        rows.append(
            {
                "key": "a" if i % 2 else "b",
                "name": "A" if i % 2 else "B",
                "source": "api_football" if i % 2 else "statsbomb",
                "drafted": 50.0 + g,
                "measured": 50.0,
                "gap": float(g),
                "baseline_gap": 0.0 if i == 0 else 30.0,
                "compared": True,
            }
        )
    rows.append({**rows[0], "gap": 0.0, "compared": False})  # not compared: never counted
    return pd.DataFrame(rows)


def test_the_verdict_needs_most_traits_within_fifteen_points():
    most = ar.calibration_summary(_table([5, 10, 15], [16, 40]))
    assert most.iloc[-1]["traits"] == 5
    assert most.iloc[-1]["within"] == pytest.approx(0.6)
    assert most.iloc[-1]["baseline_within"] == pytest.approx(0.2)
    assert most["name"].tolist()[-2:] == ["the 1 in our own data", "all 2"]
    assert ar.verdict(most)["calibrated"]
    assert "calibrated to that degree" in ar.verdict(most)["text"]
    half = ar.calibration_summary(_table([5, 10], [16, 40]))
    assert half.iloc[-1]["within"] == 0.5
    assert not ar.verdict(half)["calibrated"]  # half is not most
    assert "illustrative only" in ar.verdict(half)["text"]


def test_the_page_block_carries_the_label_traits_and_verdict():
    scale = _scale()
    own = _own(STYLE_COLUMNS)
    archs = [_archetype("pirlo", "legends", {"passes_total_per90": 99})]
    v = ar.archetype_vectors(archs, scale, own)
    info = {
        "label": ar.LABEL,
        "calibration": {"calibrated": False, "text": "Illustrative only."},
        "archetypes": [
            {
                "key": "pirlo",
                "era": "1990s",
                "drafted": "Drafted; awaits the owner's review.",
                "reviewed": False,
                "traits": [[t, 50.0, "Why."] for t in STYLE_COLUMNS],
                "sources": ["A book."],
                "calibration": None,
            }
        ],
    }
    block = archetypes_data(v, info)
    assert block["available"]
    assert block["label"] == ar.LABEL
    assert block["calibration"]["text"] == "Illustrative only."
    assert block["details"]["pirlo"]["sources"] == ["A book."]
    assert len(block["details"]["pirlo"]["traits"]) == len(STYLE_COLUMNS)
    raw = np.frombuffer(base64.b64decode(block["vectors"]), dtype=np.int8)
    assert len(raw) == len([c for c in v.columns if c.startswith("emb_")])
    assert archetypes_data(None, info) == {"available": False}
    assert similar_data(None, None, {}) == {"available": False}
