"""Fact extraction from text with one LLM call (or one per batch of statements)."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .parsers import parse_facts
from .prompts import build_batch_extraction_prompt, build_extraction_prompt

BATCH_SIZE = 10
_TURN = re.compile(r"^Speaker (.+?)\s*says : (.*)$", re.DOTALL)


def split_turn(content: str) -> Tuple[Optional[str], str]:
    """(speaker, text) of a stored turn, which reads "Speaker Alisays : ..." as
    A-Mem formats it. Content without that prefix is returned as (None, content)."""
    m = _TURN.match(content)
    return (m.group(1), m.group(2)) if m else (None, content)


class FactExtractor:
    def __init__(self, llm, temperature: float = 0.0, batch_size: int = BATCH_SIZE):
        self.llm = llm  # anything with get_completion(prompt, temperature) -> str
        self.temperature = temperature
        self.batch_size = max(1, batch_size)

    def extract(self, text: str, known_attributes: Optional[List[str]] = None,
                speaker: Optional[str] = None) -> List[dict]:
        """Facts in `text`: dicts with entity, attribute, value, valid_from, valid_until,
        temporal_status. `known_attributes` are offered to the model for reuse, and
        `speaker` tells it who "I" is."""
        if not text or not text.strip():
            return []
        prompt = build_extraction_prompt(text.strip(), known_attributes, speaker)
        return parse_facts(self.llm.get_completion(prompt, self.temperature))

    def extract_batch(self, statements: List[str],
                      known_attributes: Optional[List[str]] = None) -> List[dict]:
        """Extract from many short statements, up to `batch_size` per LLM call.

        Each returned dict also has "source": the 0-based index into
        `statements` the fact came from. A fact whose SOURCE is missing or out
        of range is dropped, unless its call held a single statement.
        """
        usable = [(i, re.sub(r"\s+", " ", s).strip())
                  for i, s in enumerate(statements) if s and s.strip()]
        facts: List[dict] = []
        for start in range(0, len(usable), self.batch_size):
            chunk = usable[start:start + self.batch_size]
            numbered = "\n".join(f"{n}. {s}" for n, (_, s) in enumerate(chunk, 1))
            prompt = build_batch_extraction_prompt(numbered, known_attributes)
            response = self.llm.get_completion(prompt, self.temperature)
            for fact in parse_facts(response, with_source=True):
                n = fact["source"]
                if n is None and len(chunk) == 1:
                    n = 1
                if n is None or not 1 <= n <= len(chunk):
                    continue
                fact["source"] = chunk[n - 1][0]
                facts.append(fact)
        return facts
