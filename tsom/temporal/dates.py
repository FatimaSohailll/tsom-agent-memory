"""Conservative date comparison for valid_from / valid_until strings.

Dates are kept as the text states them. They are compared only when both
parse cleanly (ISO "2020", "2020-05", "2020-05-17", or "17 May 2020" /
"May 17, 2020" / "May 2020"); otherwise the order is unknown.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional, Tuple

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_ISO = re.compile(r"^(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?$")
_DMY = re.compile(r"^(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?,?\s+(\d{4})$")
_MDY = re.compile(r"^([A-Za-z]{3,9})\.?\s+(?:(\d{1,2})(?:st|nd|rd|th)?,?\s+)?(\d{4})$")


def _month(name: str) -> Optional[int]:
    return _MONTHS.get(name[:3].lower())


def date_key(s: Optional[str]) -> Optional[Tuple[int, ...]]:
    """(year,), (year, month) or (year, month, day); None if not parseable."""
    if not s:
        return None
    s = s.strip()
    m = _ISO.match(s)
    if m:
        parts = tuple(int(g) for g in m.groups() if g)
        return parts if all(1 <= p for p in parts[1:]) and (len(parts) < 2 or parts[1] <= 12) else None
    m = _DMY.match(s)
    if m and _month(m.group(2)):
        return (int(m.group(3)), _month(m.group(2)), int(m.group(1)))
    m = _MDY.match(s)
    if m and _month(m.group(1)):
        key = (int(m.group(3)), _month(m.group(1)))
        return key + (int(m.group(2)),) if m.group(2) else key
    return None


def key_is_before(ka: Optional[Tuple[int, ...]], kb: Optional[Tuple[int, ...]]) -> bool:
    """True only if both keys exist and ka is strictly earlier than kb, compared
    at the coarser precision ("2020" is not before "2020-05")."""
    if ka is None or kb is None:
        return False
    k = min(len(ka), len(kb))
    return ka[:k] < kb[:k]


def is_before(a: Optional[str], b: Optional[str]) -> bool:
    """key_is_before for two date strings; unparseable dates are never "before"."""
    return key_is_before(date_key(a), date_key(b))


_YEAR = re.compile(r"\b(19|20)\d{2}\b")


def parse_observed_time(s: Optional[str]) -> Optional[Tuple[int, ...]]:
    """Date key of a session date such as "2024-03-15" or "1:56 pm on 8 May, 2023".

    None if there is no four-digit year (so labels like "facts 0-19" are never
    read as dates) or dateutil cannot parse it. Components the text does not
    contain are left out of the key, found by parsing against two different
    defaults.
    """
    if not s or not isinstance(s, str) or not _YEAR.search(s):
        return None
    try:
        from dateutil import parser
        a = parser.parse(s, fuzzy=True, default=datetime(1, 1, 1))
        b = parser.parse(s, fuzzy=True, default=datetime(2, 2, 2))
    except (ValueError, OverflowError, TypeError):
        return None
    if a.year != b.year:
        return None
    key: Tuple[int, ...] = (a.year,)
    if a.month == b.month:
        key += (a.month,)
        if a.day == b.day:
            key += (a.day,)
    return key


def effective_start(fact) -> Optional[Tuple[int, ...]]:
    """When a fact is known to have begun: its stated valid_from, else the date
    of the session it was learned in (observed_time). None if neither parses.
    Only ever used to compare, never written into a fact."""
    return date_key(fact.valid_from) or parse_observed_time(fact.observed_time)
