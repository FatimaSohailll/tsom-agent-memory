"""Session date parsing for the fact layer's validity intervals (Gap 3)."""

from __future__ import annotations

from datetime import datetime

# LoCoMo sessions look like "1:56 pm on 8 May, 2023"; the date part alone is accepted too.
FORMATS = ("%I:%M %p on %d %B, %Y", "%d %B, %Y", "%d %B %Y", "%Y-%m-%d")


def parse_session_date(text: str) -> str:
    """Return "YYYY-MM-DD", or the raw string if it is not a known format."""
    cleaned = " ".join(text.split())
    for fmt in FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return text
