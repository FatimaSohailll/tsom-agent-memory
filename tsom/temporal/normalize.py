"""Normalisation for entity/attribute/value keys."""

from __future__ import annotations

import re

_WS = re.compile(r"\s+")
_TRAILING = ".,;:!?"


def norm(s: str) -> str:
    """Lowercase, strip, collapse whitespace, strip trailing punctuation."""
    s = _WS.sub(" ", (s or "").lower()).strip()
    return s.rstrip(_TRAILING).rstrip()
