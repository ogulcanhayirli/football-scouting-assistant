"""The measured greats: peers, level, the cross source check and the page (prompt 18b)."""

import base64

import numpy as np
import pandas as pd
import pytest

from scout import greats as gr
from scout import greats_check as check
from scout.publish import OPTIONAL_SITE_FILES, SITE_FILES, publish_site
from scout.site import greats_data


def test_mid_season_is_new_year_or_midsummer_in_calendar_leagues():
    season = pd.Series([2006, 2024])
    mid = gr.mid_season(season, pd.Series([False, True]))
    assert mid.tolist() == [pd.Timestamp("2007-01-01"), pd.Timestamp("2024-07-01")]
    meta = pd.DataFrame(
        {
            "season": [2015, 2024],
            "league_key": ["la_liga", "mls"],
            "feature_date": pd.to_datetime(["2016-06-30", "2025-01-01"]),
            "age": [29.018, 37.52],
        }
    )
    age = gr.mid_season_age(meta, ["mls"])
    assert np.floor(age).tolist() == [28, 37]  # Messi: 28 in 2015/16, 37 in MLS 2024


def test_peer_standardisation_puts_regulars_on_a_regulars_scale():
    rng = np.random.default_rng(3)
    tau, k = 0.5, 900.0  # true spread, prior minutes (noise variance k * tau^2 per minute)
    minutes = np.r_[np.full(400, 150.0), np.full(40, 2000.0)]  # opponents seen twice, regulars
    x = 2.0 + rng.normal(0, tau, len(minutes)) + rng.normal(0, tau * np.sqrt(k / minutes))
    inputs = pd.DataFrame({"m": x, "gone": np.nan})
    group = pd.Series("winger", index=inputs.index)
    season = pd.Series(2006, index=inputs.index)
    priors = {"winger": {"m": k, "gone": None}}
    ref_w = {"winger": {"m": 2000 / (2000 + k), "gone": None}}
    peers = pd.Series(minutes >= 90)
    z = gr.peer_z(inputs, pd.Series(minutes), group, season, priors, ref_w, peers)
    regulars = z["m"].to_numpy()[400:]
    assert 0.7 < regulars.std() < 1.3  # a regular's spread, not inflated by the thin peers
    assert abs(regulars.mean()) < 0.4
    assert (z["gone"] == 0).all()  # no true spread or no data: the position mean
    # shrunk low minute peers sit closer to the mean than the regulars
    assert z["m"].to_numpy()[:400].std() < regulars.std()


def test_peer_ability_ranks_inside_the_peer_group_and_places_thin_rows():
    rows = pd.DataFrame(
        {
            "source_player_id": [1, 2, 3, 4],
            "season": 2006,
            "minutes": [900.0, 900.0, 900.0, 40.0],
            "sub_position": "Left Winger",
        }
    )
    z = pd.DataFrame({"goals_per90": [1.0, 0.0, -1.0, 0.5]})
    weights = pd.DataFrame({"position": ["winger"], "measure": ["goals_per90"], "weight": [1.0]})
    peers = rows["minutes"] >= 90
    out = gr.peer_ability(z, rows, weights, peers)
    assert out["ability_pct_1s"].round(1).tolist() == [100.0, 66.7, 33.3, 66.7]
    assert out["ability_pct"].notna().all()


def test_level_curves_take_the_level_out():
    rng = np.random.default_rng(5)
    n = 3000
    ability = pd.DataFrame({"ability_pct": rng.uniform(0, 100, n)})
    ability["ability_pct_1s"] = ability["ability_pct"]
    z = np.c_[0.03 * ability["ability_pct"] + rng.normal(0, 1, n), rng.normal(0, 1, n)]
    group = pd.Series("winger", index=ability.index)
    curves = gr.level_curves(z, ability, group)
    out = curves.apply(z, ability, group)
    assert abs(np.corrcoef(out[:, 0], ability["ability_pct"])[0, 1]) < 0.05
    assert np.corrcoef(z[:, 0], ability["ability_pct"])[0, 1] > 0.5


def _side(fixture, team, name, home, goals, players, kickoff="2016-08-20 18:00"):
    return pd.DataFrame(
        {
            "fixture_id": fixture,
            "kickoff": pd.Timestamp(kickoff, tz="UTC"),
            "team_id": team,
            "team_name": name,
            "is_home": home,
            "team_goals": goals[0],
            "opponent_goals": goals[1],
            "season": 2016,
            "league_key": "la_liga",
            "source_player_id": [p[0] for p in players],
            "player_name": [p[1] for p in players],
            "shirt_number": [p[2] for p in players],
            "minutes": 90.0,
        }
    )


def _two_sources():
    rng = np.random.default_rng(1)
    sb = pd.concat(
        [
            _side(
                1,
                217,
                "Barcelona",
                True,
                (2, 1),
                [
                    (5503, "Lionel Andrés Messi Cuccittini", 10),
                    (20131, "Xavier Hernández Creus", 6),
                ],
            ),
            _side(1, 220, "Real Madrid", False, (1, 2), [(5207, "Sergio Ramos García", 4)]),
        ],
        ignore_index=True,
    )
    sb["player_nickname"] = ["Lionel Messi", "Xavi", None]
    af = pd.concat(
        [
            _side(900, 529, "Barcelona", True, (2, 1), [(154, "L. Messi", 10), (751, "Xavi", 6)]),
            _side(900, 541, "Real Madrid", False, (1, 2), [(738, "Sergio Ramos", 4)]),
        ],
        ignore_index=True,
    )
    for m in check.COMPARED[1:]:
        values = rng.integers(0, 6, len(sb)).astype(float)
        sb[m] = values
        af[m] = values if m != "interceptions" else values[::-1]
    sb["passes_accurate"] = sb["passes_total"]
    return sb, af


def test_matches_and_players_are_linked_across_sources():
    sb, af = _two_sources()
    matches = check.link_matches(sb, af)
    assert matches[["sb_fixture", "af_fixture"]].values.tolist() == [[1, 900]]
    links = check.link_players(sb, af, matches)
    assert dict(zip(links["sb_player"], links["af_player"], strict=True)) == {
        5503: 154,
        20131: 751,
        5207: 738,
    }
    ids = check.player_ids(links)
    assert ids[5503] == 154
    pairs = check.paired_rows(sb, af, links)
    table = check.measure_table(pairs).set_index("measure")
    assert table.loc["goals", "correlation"] == pytest.approx(1.0)
    assert table.loc["goals", "mean_difference"] == 0


def test_a_measure_below_the_floor_is_excluded():
    table = pd.DataFrame(
        {
            "measure": ["tackles_total", "interceptions", "duels_won", "duels_total"],
            "correlation": [0.62, 0.9, 0.81, 0.69],
        }
    )
    out = check.excluded_measures(table)
    assert out["tackles_total_per90"] == ["tackles_total"]
    assert out["duel_win_rate"] == ["duels_total"]
    assert out["share_duels"] == ["duels_total"]
    assert "interceptions_per90" not in out
    assert "duels_won_per90" not in out


def _greats_table():
    emb = np.eye(3)[[0, 1]]
    return pd.DataFrame(
        {
            "great": ["Lionel Messi", "Lionel Messi"],
            "source": ["statsbomb", "api_football"],
            "season": [2006, 2015],
            "age": [19.5, 28.5],
            "age_years": [19, 28],
            "team": ["Barcelona", "Barcelona"],
            "league_key": ["la_liga", "la_liga"],
            "group": ["winger", "winger"],
            "minutes": [2001.0, 2728.0],
            "ability_pct": [96.0, 100.0],
            "peers": ["Barcelona and its opponents", "every player of his league"],
            "thin": [False, False],
            "emb_0": emb[:, 0],
            "emb_1": emb[:, 1],
            "emb_2": emb[:, 2],
        }
    )


def test_nearest_to_great_searches_today_of_that_age_and_group():
    today = pd.DataFrame(
        {
            "source_player_id": [1, 2, 3, 4],
            "player_name": ["Near", "Far", "Older", "Keeper"],
            "age_mid": [19.2, 19.9, 20.1, 19.5],
            "group": ["winger", "winger", "winger", "goalkeeper"],
            "emb_0": [0.9, 0.0, 1.0, 1.0],
            "emb_1": [0.1, 1.0, 0.0, 0.0],
            "emb_2": [0.0, 0.0, 0.0, 0.0],
        }
    )
    query, found = gr.nearest_to_great(_greats_table(), today, "Lionel Messi", 19)
    assert query["season"] == 2006
    assert found["player_name"].tolist() == ["Near", "Far"]  # 19 years old wingers only
    _, pool = gr.nearest_to_great(_greats_table(), today, "Lionel Messi", 19, only={2})
    assert pool["player_name"].tolist() == ["Far"]
    with pytest.raises(KeyError):
        gr.nearest_to_great(_greats_table(), today, "Lionel Messi", 40)


def test_the_page_block_holds_vectors_label_and_attribution():
    info = {"label": gr.LABEL, "attribution": gr.ATTRIBUTION, "excluded": ["tackles_total_per90"]}
    block = greats_data(_greats_table(), info, {"la_liga": "La Liga"})
    assert block["available"]
    assert block["label"].startswith("Style similarity")
    assert "StatsBomb" in block["attribution"]
    assert block["logo"] == "statsbomb-logo.png"
    rows = [dict(zip(block["columns"], r, strict=True)) for r in block["rows"]]
    assert rows[0]["league"] == "La Liga"
    assert rows[0]["age"] == 19
    raw = np.frombuffer(base64.b64decode(block["vectors"]), dtype=np.int8).reshape(2, 3)
    assert raw[0].tolist() == [127, 0, 0]
    assert greats_data(None, info, {}) == {"available": False}


class FakeS3:
    def __init__(self):
        self.keys = []

    def put_object(self, Bucket, Key, Body, **kw):  # noqa: N803 (boto3's names)
        self.keys.append(Key)


def test_the_logo_is_published_only_when_the_site_has_it(tmp_path):
    for key in SITE_FILES:
        (tmp_path / key).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / key).write_bytes(b"x")
    s3 = FakeS3()
    publish_site(s3, "site", tmp_path)
    assert "statsbomb-logo.png" not in s3.keys
    (tmp_path / "statsbomb-logo.png").write_bytes(b"png")
    s3 = FakeS3()
    publish_site(s3, "site", tmp_path)
    assert set(OPTIONAL_SITE_FILES) <= set(s3.keys)
    assert s3.keys[-1] == "index.html"
