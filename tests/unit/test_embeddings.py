"""Style vectors, the learned embedding, the tests that decide and the queries
(decision 10, prompt 18)."""

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scout import cli
from scout import embeddings as emb
from scout.scouting import COMPOSITE_CANDIDATES
from scout.site import HISTORY_TOP, similar_data

GROUPS = {"Centre-Forward": "centre_forward", "Centre-Back": "centre_back"}


def synthetic(players=80, seasons=range(2015, 2026), seed=0):
    """Feature rows of players whose stats come from a fixed style (their
    own), a level (shared direction) and noise; the ability percentile is
    the level's rank. Returns (features, ability)."""
    rng = np.random.default_rng(seed)
    stats = [
        *COMPOSITE_CANDIDATES,
        "shots_total",
        "dribbles_attempts",
        "passes_total",
        "duels_total",
    ]
    loading = rng.normal(0, 1, len(stats))
    rows = []
    for g, position in enumerate(GROUPS):
        style = rng.normal(0, 1, (players, len(stats)))
        level = rng.uniform(0, 1, players)
        for season in seasons:
            for p in range(players):
                pid = 1000 * g + p
                values = 3 + 0.6 * style[p] + 1.0 * level[p] * loading
                values = np.clip(values + rng.normal(0, 0.25, len(stats)), 0.01, None)
                row = dict(zip(stats, values, strict=True))
                row.update(
                    league_key="lg_a" if p % 2 else "lg_b",
                    season=season,
                    source_player_id=pid,
                    player_name=f"Player {pid}",
                    feature_date=pd.Timestamp(f"{season + 1}-06-30"),
                    age=18 + p % 15 + (season - 2015) + 0.5,
                    team_id=10 * g + p % 4,
                    team_name=f"Club {p % 4}",
                    sub_position=position,
                    tier="match",
                    regular=True,
                    minutes=900 + 60 * (p % 30),
                    market_value=1e6 * (1 + level[p]),
                    level=level[p],
                )
                rows.append(row)
    features = pd.DataFrame(rows)
    features["minutes_share"] = features["minutes"] / 3420
    keys = [features["sub_position"], features["season"]]
    pct = features["level"].groupby(keys).rank(pct=True) * 100
    ability = pd.DataFrame(
        {"ability_pct": pct, "ability_pct_1s": (pct + rng.normal(0, 5, len(pct))).clip(0, 100)}
    )
    features["value_rank"] = features.groupby("season")["market_value"].rank(ascending=False)
    return features.drop(columns="level"), ability


@pytest.fixture(scope="module")
def built():
    features, ability = synthetic()
    return features, ability, emb.style_vectors(features, ability)


# Style vectors


def test_on_pitch_shares_divide_the_team_share_by_the_minutes_share():
    f = pd.DataFrame(
        {
            "league_key": "x",
            "season": 2020,
            "team_id": [1, 1, 2],
            "tier": "match",
            "minutes": [1710.0, 1710.0, 900.0],
            "minutes_share": [0.5, 0.5, 0.25],
            "shots_total": [30.0, 10.0, None],
            "dribbles_attempts": [5.0, 5.0, 2.0],
            "passes_total": [100.0, 300.0, 50.0],
            "duels_total": [None, 20.0, 4.0],
            **{c: 1.0 for c in COMPOSITE_CANDIDATES},
        }
    )
    x = emb.style_inputs(f)
    assert x.loc[0, "share_shots"] == pytest.approx(30 / 40 / 0.5)
    assert x.loc[1, "share_passes"] == pytest.approx(300 / 400 / 0.5)
    assert x.loc[0, "share_duels"] == 0.0  # no entry, but his team's duels are recorded
    assert np.isnan(x.loc[2, "share_shots"])  # nobody in his team has shots: unknown
    assert list(x.columns) == emb.STYLE_COLUMNS


def test_shrinkage_pulls_low_minutes_further_and_standardises():
    inputs = pd.DataFrame({"a": [2.0, 2.0, 0.0, 0.0, 1.0, 1.0], "b": 1.0})
    minutes = pd.Series([500.0, 3000.0, 500.0, 3000.0, 1000.0, 1000.0])
    group = pd.Series(["g"] * 6)
    season = pd.Series([2020] * 6)
    z = emb.shrink_standardise(inputs, minutes, group, season, {"g": {"a": 1000.0, "b": None}})
    assert abs(z.loc[0, "a"]) < abs(z.loc[1, "a"])  # same raw value, fewer minutes
    assert z["a"].mean() == pytest.approx(0.0)
    assert z["a"].std() == pytest.approx(1.0)
    assert (z["b"] == 0).all()  # no prior: no spread, the mean


def test_level_removal_leaves_no_level_in_the_mean_or_the_spread():
    rng = np.random.default_rng(1)
    n = 2000
    ability = pd.DataFrame({"ability_pct": rng.uniform(0, 100, n)})
    ability["ability_pct_1s"] = ability["ability_pct"]
    a = ability["ability_pct"].to_numpy() / 100
    values = np.column_stack(
        [3 * a + rng.normal(0, 1, n), rng.normal(0, 1, n) * (0.2 + 2 * a)]  # mean, spread
    )
    out = emb.remove_level(values, ability, pd.Series(["g"] * n), pd.Series([2020] * n))
    assert abs(np.corrcoef(out[:, 0], a)[0, 1]) < 0.05
    assert np.corrcoef(values[:, 1] ** 2, a)[0, 1] > 0.3
    assert abs(np.corrcoef(out[:, 1] ** 2, a)[0, 1]) < 0.1


def test_neighbours_are_unrelated_in_level_once_it_is_removed(built):
    features, ability, sv = built
    kept = emb.style_vectors(features, ability, priors=sv.priors, keep_level=True)
    before = emb.level_check(kept.style, kept.meta, range(2015, 2026))
    after = emb.level_check(sv.style, sv.meta, range(2015, 2026))
    assert before["pearson"] > 0.3
    assert abs(after["pearson"]) < 0.1


def test_vector_rows_are_regular_match_rows_with_a_group_from_2015():
    features, ability = synthetic(players=40, seasons=range(2014, 2017))
    features.loc[0, "regular"] = False
    features.loc[1, "tier"] = "season"
    features.loc[2, "sub_position"] = None
    rows = emb.vector_rows(features, ability)
    assert not rows[[0, 1, 2]].any()
    assert not rows[features["season"] == 2014].any()
    assert rows[(features["season"] >= 2015) & (features.index > 2)].all()


def test_main_rows_keep_the_league_with_most_minutes():
    meta = pd.DataFrame(
        {
            "source_player_id": [1, 1, 2],
            "season": 2020,
            "league_key": ["a", "b", "a"],
            "minutes": [600.0, 1500.0, 900.0],
        }
    )
    assert emb.main_rows(meta).tolist() == [False, True, True]


# The learned embedding


def test_info_nce_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    encoder = emb.init_encoder(5, 4, 6, rng)
    xa = rng.normal(size=(7, 5))
    xb = xa + 0.3 * rng.normal(size=(7, 5))
    _, grads = emb.info_nce(encoder, xa, xb, 0.2)
    eps = 1e-6
    for param, grad in zip(encoder.params(), grads, strict=True):
        flat, g = param.reshape(-1), grad.reshape(-1)
        for i in range(0, flat.size, 3):
            old = flat[i]
            flat[i] = old + eps
            plus, _ = emb.info_nce(encoder, xa, xb, 0.2)
            flat[i] = old - eps
            minus, _ = emb.info_nce(encoder, xa, xb, 0.2)
            flat[i] = old
            assert (plus - minus) / (2 * eps) == pytest.approx(g[i], abs=1e-7)


def test_training_lowers_the_loss_and_finds_the_same_player(built):
    _, _, sv = built
    pairs = emb.season_pairs(sv.meta)
    assert set(pairs["group"]) == set(GROUPS.values())
    encoder, history = emb.train_encoder(sv.style, pairs, dim=8, temperature=0.1, epochs=8)
    assert history[-1] < history[0]
    vectors = encoder.embed(sv.style)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    found = emb.identity_retrieval(vectors, sv.meta, (2024, 2025))
    assert found["hit"].mean() > 5 * found["random"].mean()


def test_identity_retrieval_ranks_the_hidden_season_both_ways():
    meta = pd.DataFrame(
        {
            "source_player_id": [1, 2, 3, 1, 2, 3],
            "season": [2023, 2023, 2023, 2024, 2024, 2024],
            "group": "g",
            "main": True,
        }
    )
    same = np.eye(3)[[0, 1, 2, 0, 1, 2]]  # each player keeps his direction
    found = emb.identity_retrieval(same, meta, (2023, 2024), k=1)
    assert len(found) == 6
    assert found["hit"].all()
    swapped = np.eye(3)[[0, 1, 2, 1, 0, 2]]  # players 1 and 2 swap styles
    found = emb.identity_retrieval(swapped, meta, (2023, 2024), k=1)
    assert found["hit"].sum() == 2  # only player 3, both ways
    assert found["random"].iloc[0] == pytest.approx(1 / 3)


def test_stored_embedding_round_trips(built):
    _, _, sv = built
    encoder, _ = emb.train_encoder(sv.style, emb.season_pairs(sv.meta), 8, 0.1, epochs=1)
    stored = emb.StoredEmbedding(sv.columns, sv.priors, "learned", encoder, {"dim": 8})
    back = emb.StoredEmbedding.from_json(stored.to_json())
    assert back.method == "learned"
    assert back.meta == {"dim": 8}
    assert back.priors == json.loads(json.dumps(sv.priors))
    assert np.allclose(back.vectors(sv.style), stored.vectors(sv.style), atol=1e-6)
    plain = emb.StoredEmbedding.from_json(
        emb.StoredEmbedding(sv.columns, sv.priors, "baseline", None, {}).to_json()
    )
    assert plain.encoder is None
    assert np.allclose(plain.vectors(sv.style), emb.unit(sv.style))


# The replacement check


def test_replacement_pairs_pair_a_sale_with_a_same_position_signing():
    meta = pd.DataFrame(
        {
            "source_player_id": [1, 2, 3, 4, 5],
            "season": 2022,
            "feature_date": pd.Timestamp("2023-06-30"),
            "group": ["winger", "winger", "centre_back", "winger", "winger"],
            "main": True,
        }
    )
    moves = pd.DataFrame(
        {
            "source_player_id": [1, 2, 3, 4, 5],
            "transfer_date": pd.to_datetime(
                ["2023-07-10", "2023-08-20", "2023-07-15", "2023-07-20", "2023-12-01"]
            ),
            "team_out_id": [7, 8, 9, 10, 11],
            "team_out_name": "x",
            "team_in_id": [50, 7, 7, 7, 7],
            "team_in_name": "y",
            "kind": ["transfer", "free", "transfer", "loan_return", "loan"],
        }
    )
    pairs = emb.replacement_pairs(moves, meta)
    # player 2 (41 days later, same group) only: 3 is a centre back, 4 a
    # loan return, 5 arrives 144 days later
    assert pairs[["pid_out", "pid_in"]].values.tolist() == [[1, 2]]
    ranks = emb.replacement_ranks(np.eye(5)[[0, 0, 1, 2, 3]], meta, pairs)
    assert ranks["rank"].tolist() == [1]  # the signing is the nearest
    assert ranks["pool"].tolist() == [3]  # wingers of his season, the departed left out


def test_a_vector_older_than_the_limit_is_not_used():
    meta = pd.DataFrame(
        {
            "source_player_id": [1],
            "season": 2020,
            "feature_date": pd.Timestamp("2021-06-30"),
            "main": True,
        }
    )
    moves = pd.DataFrame(
        {"pid": [1, 1], "transfer_date": pd.to_datetime(["2021-08-01", "2023-08-01"])}
    )
    assert emb.vector_at(meta, moves, "transfer_date").tolist() == [0, -1]


# Queries


def _index():
    meta = pd.DataFrame(
        {
            "source_player_id": [1, 1, 2, 3, 4, 5, 6],
            "player_name": ["K. Yıldız", "K. Yıldız", "K. Yildirim", "A", "B", "C", "D"],
            "season": [2023, 2025, 2025, 2025, 2025, 2025, 2025],
            "league_key": ["serie_a", "serie_a", "super_lig", "epl", "epl", "la_liga", "epl"],
            "group": ["winger", "winger", "winger", "winger", "winger", "winger", "centre_back"],
            "age": [19.2, 21.2, 25.0, 19.5, 22.9, 20.1, 21.0],
            "ability_pct": [80.0, 98.0, 40.0, 90.0, 30.0, 70.0, 50.0],
            "value_rank": [400.0, 20.0, 900.0, 300.0, 2000.0, 800.0, 100.0],
            "team_name": "x",
            "main": True,
        }
    )
    v = np.array(
        [[1, 0.1], [0.2, 1], [1, 0.3], [1, 0], [0.3, 1], [0.6, 0.8], [0.2, 1]], dtype=float
    )
    return emb.SimilarIndex(meta, v)


def test_find_ignores_accents_and_prefers_the_exact_name():
    index = _index()
    assert index.find("k. yildiz") == 1
    assert index.find("yildi") == 1  # two match; the higher value rank wins
    assert index.find("Kenan Yıldız") == 1  # the surname and the initial
    assert index.find("Kenan Yildirim") == 2
    assert index.find(2) == 2
    with pytest.raises(KeyError):
        index.find("nobody")


def test_similar_now_searches_today_in_the_same_group_with_filters():
    index = _index()
    query, found = index.similar_now(1)
    assert query["season"] == 2025
    assert found["source_player_id"].tolist()[:2] == [4, 5]  # direction (0.2, 1)
    assert 6 not in found["source_player_id"].tolist()  # a centre back
    assert 1 not in found["source_player_id"].tolist()
    _, young = index.similar_now(1, age=(None, 21))
    assert set(young["source_player_id"]) == {3, 5}
    _, good = index.similar_now(1, ability=(60, None), leagues=["epl", "la_liga"])
    assert set(good["source_player_id"]) == {3, 5}


def test_looks_like_takes_his_season_at_that_age_against_today_a_year_either_side():
    index = _index()
    query, found = index.looks_like(1, 19)
    assert query["season"] == 2023  # aged 19 then
    assert set(found["source_player_id"]) == {3, 5}  # aged 18 to 20 today, wingers
    assert found["source_player_id"].iloc[0] == 3  # direction (1, 0.1)
    query, _ = index.looks_like(1, 30)  # no season at 30: the nearest age
    assert query["season"] == 2025


def test_current_neighbours_are_today_same_group_without_himself():
    table = emb.current_neighbours(_index(), k=2)
    assert set(table["season"]) == {2025}
    assert (table["player_id"] != table["neighbour_id"]).all()
    assert table.groupby("player_id").size().max() == 2
    assert table.loc[table["player_id"] == 6, "neighbour_id"].empty  # alone in his group
    assert set(table.columns) <= set(emb.DICTIONARY["neighbours"])


def test_the_dictionary_names_every_column(built):
    _, _, sv = built
    encoder, _ = emb.train_encoder(sv.style, emb.season_pairs(sv.meta), 8, 0.1, epochs=1)
    frame = emb.vectors_frame(
        sv, emb.StoredEmbedding(sv.columns, sv.priors, "learned", encoder, {})
    )
    words = emb.DICTIONARY["vectors"]
    for column in frame.columns:
        assert column in words or f"{column.split('_')[0]}_*" in words, column


# The page's file


def test_similar_data_holds_today_and_the_past_of_once_top_players(built):
    _, _, sv = built
    stored = emb.StoredEmbedding(sv.columns, sv.priors, "baseline", None, {})
    frame = emb.vectors_frame(sv, stored)
    current = pd.DataFrame(
        {
            "source_player_id": [0, 1],
            "current_club": ["Club 0", "Elsewhere"],
            "current_club_id": [0, 999],  # player 0 stays at team 0, player 1 moved
        }
    )
    data = similar_data(frame, current, {"lg_a": "League A"}, {"learned": 0.2})
    assert data["available"]
    assert data["season"] == 2025
    assert data["method"] == "baseline"
    today = [dict(zip(data["today"]["columns"], r, strict=True)) for r in data["today"]["rows"]]
    assert len(today) == int((sv.meta["main"] & (sv.meta["season"] == 2025)).sum())
    by_id = {o["id"]: o for o in today}
    assert by_id[0]["club_now"] is None
    assert data["clubs"][by_id[1]["club_now"]] == "Elsewhere"
    assert data["leagues"][by_id[1]["league"]] == "League A"
    past = [dict(zip(data["history"]["columns"], r, strict=True)) for r in data["history"]["rows"]]
    top = set(sv.meta.loc[sv.meta["value_rank"] <= HISTORY_TOP, "source_player_id"])
    assert {o["id"] for o in past} <= top
    assert all(o["season"] < 2025 for o in past)
    raw = np.frombuffer(base64.b64decode(data["today"]["vectors"]), dtype=np.int8)
    raw = raw.reshape(len(today), data["dim"])
    norms = np.linalg.norm(raw / data["scale"], axis=1)
    assert np.allclose(norms, 1.0, atol=0.05)
    assert similar_data(None, None, {}) == {"available": False}
    json.dumps(data, allow_nan=False)


# The commands, end to end


def _write_data(tmp_path):
    features, ability = synthetic()
    processed = tmp_path / "processed"
    for (league, season), part in features.groupby(["league_key", "season"]):
        folder = processed / "features" / "player_season_features"
        folder = folder / f"league={league}" / f"season={season}"
        folder.mkdir(parents=True)
        part.drop(columns=["season", "value_rank"]).to_parquet(folder / "data.parquet")
    scouting = pd.concat(
        [features[["league_key", "season", "source_player_id", "value_rank"]], ability], axis=1
    )
    (processed / "models").mkdir(parents=True)
    scouting.to_parquet(processed / "models" / "scouting.parquet")
    pid = features["source_player_id"]
    sold = features[(features["season"] == 2023) & (pid % 7 == 0)]
    transfers = pd.DataFrame(
        {
            "source_player_id": sold["source_player_id"].to_numpy(),
            "transfer_date": pd.Timestamp("2024-07-15"),
            "team_out_id": sold["team_id"].to_numpy(),
            "team_out_name": "x",
            "team_in_id": 99,
            "team_in_name": "y",
            "kind": "transfer",
        }
    )
    signed = transfers.assign(
        source_player_id=transfers["source_player_id"] + 1,
        team_in_id=transfers["team_out_id"],
        team_out_id=98,
        kind="free",
    )
    (processed / "transfers").mkdir()
    pd.concat([transfers, signed]).to_parquet(processed / "transfers" / "transfers.parquet")
    template = Path(__file__).parents[2] / "configs" / "backfill.free.yaml"
    settings = yaml.safe_load(template.read_text(encoding="utf-8"))
    settings["paths"] = {
        "state_dir": str(tmp_path / "state"),
        "processed_dir": str(processed),
        "reports_dir": str(tmp_path / "reports"),
    }
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(settings), encoding="utf-8")
    return config, processed


def test_train_embed_and_similar_from_the_command_line(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(emb, "DIMENSIONS", (8,))
    monkeypatch.setattr(emb, "TEMPERATURES", (0.1,))
    config, processed = _write_data(tmp_path)
    assert cli.main(["--config", str(config), "embed", "--train"]) == 0
    report = (tmp_path / "reports" / "embeddings.md").read_text()
    assert "## Identity retrieval, 2023 to 2025" in report
    assert "## Replacement check" in report
    stored = json.loads((processed / "models" / "stored" / emb.STORED).read_text())
    assert stored["method"] in ("learned", "baseline")
    assert stored["adopted"] == (stored["tested"]["learned"] > stored["tested"]["baseline"])
    first = pd.read_parquet(processed / "embeddings" / "neighbours.parquet")

    assert cli.main(["--config", str(config), "embed"]) == 0  # the weekly run: no training
    again = pd.read_parquet(processed / "embeddings" / "neighbours.parquet")
    assert again[["player_id", "rank", "neighbour_id"]].equals(
        first[["player_id", "rank", "neighbour_id"]]
    )
    capsys.readouterr()
    assert cli.main(["--config", str(config), "similar", "Player 7", "--age-max", "40"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Playing like Player 7 now")
    assert out.count("\n| ") == 11  # header and ten players
    assert cli.main(["--config", str(config), "similar", "Player 7", "--at-age", "25"]) == 0
    assert "who look like Player 7 at 25" in capsys.readouterr().out
