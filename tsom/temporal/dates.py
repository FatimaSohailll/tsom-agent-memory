"""Conservative date comparison for valid_from / valid_until strings.

Dates are kept as the text states them. They are compared only when both
parse cleanly (ISO "2020", "2020-05", "2020-05-17", or "17 May 2020" /
"May 17, 2020" / "May 2020"); otherwise the order is unknown.
"""

from __future__ import annotations

import re
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


def is_before(a: Optional[str], b: Optional[str]) -> bool:
    """True only if both parse and a is strictly earlier than b, compared at
    the precision both share ("2020" is not before "2020-05")."""
    ka, kb = date_key(a), date_key(b)
    if ka is None or kb is None:
        return False
    k = min(len(ka), len(kb))
    return ka[:k] < kb[:k]
