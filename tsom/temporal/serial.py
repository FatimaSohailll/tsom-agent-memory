"""Serial-numbered fact lines, e.g. "12. The capital of France is Paris."."""

from __future__ import annotations

import re
from typing import List, Tuple

_SERIAL = re.compile(r"^\s*(\d+)\.\s+(.*)")


def split_serial_facts(text: str) -> List[Tuple[int, str]]:
    """Return (serial_number, statement) for each numbered line, in order.

    The serial number is used as observed_at (FactConsolidation). Lines that
    are not numbered are ignored.
    """
    out: List[Tuple[int, str]] = []
    for line in (text or "").splitlines():
        m = _SERIAL.match(line)
        if m and m.group(2).strip():
            out.append((int(m.group(1)), m.group(2).strip()))
    return out
