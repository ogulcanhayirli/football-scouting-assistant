import pandas as pd
import pytest

from scout.matching import match_players, normalize_club, normalize_name


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("İsmail Yılmazer", "ismail yilmazer"),
        ("Çağlar Söyüncü", "caglar soyuncu"),
        ("ŞAHİNOĞLU", "sahinoglu"),
        ("M. Ødegaard", "m odegaard"),
        ("Jean-Philippe Mateta", "jean philippe mateta"),
        (None, ""),
    ],
)
def test_normalize_name(raw, expected):
    assert normalize_name(raw) == expected


def test_normalize_club_strips_suffixes():
    assert normalize_club("Test Town FC") == "test town"


def af(pid, name, first, last, dob, team=""):
    return {
        "source_player_id": pid,
        "player_name": name,
        "firstname": first,
        "lastname": last,
        "birth_date": dob,
        "team_name": team,
    }


def tm(pid, name, dob, clubs=""):
    return {"source_player_id": pid, "player_name": name, "birth_date": dob, "club_names": clubs}


def test_matches_abbreviated_and_turkish_names():
    stats = pd.DataFrame(
        [
            af(1, "M. Salah", "Mohamed Salah Hamed", "Mahrous Ghaly", "1992-06-15"),
            af(2, "İ. Yılmazer", "İsmail", "Yılmazer", "1999-05-21"),
        ]
    )
    vals = pd.DataFrame(
        [tm(10, "Mohamed Salah", "1992-06-15"), tm(20, "Ismail Yilmazer", "1999-05-21")]
    )
    result = match_players(stats, vals).set_index("af_player_id")
    assert result.loc[1, "tm_player_id"] == 10
    assert result.loc[2, "tm_player_id"] == 20
    assert set(result["reason"]) == {"matched"}


def test_reasons_for_non_matches():
    stats = pd.DataFrame(
        [
            af(1, "A. Nobody", "Alex", "Nobody", None),
            af(2, "B. Other", "Ben", "Other", "2000-01-01"),
            af(3, "C. Wrong", "Chris", "Wrong", "2001-01-01"),
        ]
    )
    vals = pd.DataFrame([tm(30, "Completely Different", "2001-01-01")])
    reasons = match_players(stats, vals).set_index("af_player_id")["reason"].to_dict()
    assert reasons == {1: "no_birth_date", 2: "no_candidate_same_dob", 3: "low_score"}


def test_club_breaks_ties_between_namesakes():
    stats = pd.DataFrame([af(1, "Fred", "Fred", "", "1993-03-05", team="Fenerbahçe")])
    vals = pd.DataFrame(
        [
            tm(40, "Fred", "1993-03-05", clubs="Manchester United Fenerbahce"),
            tm(41, "Fred", "1993-03-05", clubs="Some Other Club"),
        ]
    )
    result = match_players(stats, vals).iloc[0]
    assert result["tm_player_id"] == 40


def test_identical_candidates_without_club_are_ambiguous():
    stats = pd.DataFrame([af(1, "Fred", "Fred", "", "1993-03-05")])
    vals = pd.DataFrame([tm(40, "Fred", "1993-03-05"), tm(41, "Fred", "1993-03-05")])
    assert match_players(stats, vals).iloc[0]["reason"] == "ambiguous"
