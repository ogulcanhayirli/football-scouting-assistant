r"""Names as API Football sends them, repaired (rule 7, prompt 18b).

Two source errors, both seen in cached responses and both lossless to undo:

* **UTF-8 read as Latin-1 or Windows-1252** ("O. H\u00c3\u00b8jlund" for
  O. Højlund, "A. Avdi\u00c4\u2021" for A. Avdić; in "O. Çakıroğlu" the ı
  arrived right and the rest did not). Each run of a lead character
  (Â to ô, the bytes C2 to F4) and its continuation characters is turned
  back into its bytes and decoded as UTF-8; a run that does not decode, or
  decodes to anything but a letter, is left as it is, so a real name is
  never touched.
* **HTML entities** ("M. O&apos;Riley", "Jo&#227;o Marcelo") are unescaped.

741 distinct names in 379 of 1,986 processed partitions were affected on 25
October 2026, most of them the ``&apos;`` of names like N'Diaye and O'Brien
(``scripts/repair_names.py``). The contracts reject a name that
``clean_name`` would still change.
"""

from __future__ import annotations

import contextlib
import html
import re
import unicodedata

import pandas as pd

# Characters Windows-1252 puts on the bytes 80 to 9F (Latin-1 has control codes there).
_CP1252: dict[str, int] = {}
for _b in range(0x80, 0xA0):
    with contextlib.suppress(UnicodeDecodeError):  # five bytes are unassigned
        _CP1252[bytes([_b]).decode("cp1252")] = _b
_CONTINUATION = "\u0080-\u00bf" + "".join(re.escape(c) for c in _CP1252)
MOJIBAKE = re.compile(f"[\u00c2-\u00f4][{_CONTINUATION}]{{1,3}}")
ENTITY = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")
# Every name column of the API Football tables (each frame builder repairs the ones it has).
NAME_COLUMNS = [
    "player_name",
    "firstname",
    "lastname",
    "assist_name",
    "coach_name",
    "team_name",
    "opponent_name",
    "team_out_name",
    "team_in_name",
]


def _byte(ch: str) -> int:
    return ord(ch) if ord(ch) < 256 else _CP1252[ch]


def _redecode(match: re.Match) -> str:
    text = match.group()
    raw = bytes(_byte(c) for c in text)
    need = 1 if raw[0] < 0xE0 else 2 if raw[0] < 0xF0 else 3
    if len(raw) <= need:
        return text
    try:
        char = raw[: need + 1].decode("utf-8")
    except UnicodeDecodeError:
        return text
    if not all(unicodedata.category(c)[0] in "LM" for c in char):
        return text
    return char + text[need + 1 :]


def clean_name(value):
    """The name with HTML entities unescaped and UTF-8 mojibake decoded;
    anything else (missing values, clean names) unchanged."""
    if not isinstance(value, str):
        return value
    text = html.unescape(value) if ENTITY.search(value) else value
    return MOJIBAKE.sub(_redecode, text) if MOJIBAKE.search(text) else text


def garbled(values: pd.Series) -> pd.Series:
    """True where ``clean_name`` would change the name (the contract's check)."""
    text = values.astype("string")
    suspect = text.str.contains(ENTITY, na=False) | text.str.contains(MOJIBAKE, na=False)
    out = pd.Series(False, index=values.index)
    if suspect.any():
        out[suspect] = [clean_name(v) != v for v in values[suspect]]
    return out


def clean_names(frame: pd.DataFrame, columns=NAME_COLUMNS, log=None) -> pd.DataFrame:
    """``clean_name`` on the name columns that exist; repairs logged."""
    for column in columns:
        if column not in frame:
            continue
        bad = garbled(frame[column])
        if bad.any():
            fixed = frame.loc[bad, column].map(clean_name)
            if log is not None:
                pairs = sorted(set(zip(frame.loc[bad, column], fixed, strict=True)))
                log.warning(
                    "%d %s values garbled by the source, repaired (%s)",
                    int(bad.sum()),
                    column,
                    "; ".join(f"{a!r} -> {b!r}" for a, b in pairs[:3])
                    + ("; ..." if len(pairs) > 3 else ""),
                )
            frame.loc[bad, column] = fixed
    return frame
