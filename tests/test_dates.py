import json
import re
from pathlib import Path

from tsom.dates import parse_session_date

DATA = Path(__file__).resolve().parent.parent / "data" / "locomo10.json"


def test_every_locomo_session_date_parses():
    dates = [v for sample in json.loads(DATA.read_text())
             for key, v in sample["conversation"].items() if key.endswith("date_time")]
    assert len(dates) == 288
    parsed = [parse_session_date(d) for d in dates]
    assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", p) for p in parsed), \
        [d for d, p in zip(dates, parsed) if p == d]


def test_known_formats():
    assert parse_session_date("1:56 pm on 8 May, 2023") == "2023-05-08"
    assert parse_session_date("6:30 pm on 14 June, 2023") == "2023-06-14"
    assert parse_session_date("10:37 am on 27 June, 2023") == "2023-06-27"
    assert parse_session_date("12:09 am on 13 September, 2023") == "2023-09-13"
    assert parse_session_date("8 May, 2023") == "2023-05-08"
    assert parse_session_date("2023-05-08") == "2023-05-08"


def test_unknown_format_falls_back_to_raw():
    assert parse_session_date("last Tuesday") == "last Tuesday"
    assert parse_session_date("d1") == "d1"
