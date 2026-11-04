"""Fact extraction from text with one LLM call (or one per batch of statements)."""

from __future__ import annotations

import re
from typing import List

from .parsers import parse_facts
from .prompts import FACT_BATCH_EXTRACTION_PROMPT, FACT_EXTRACTION_PROMPT

BATCH_SIZE = 10


class FactExtractor:
    def __init__(self, llm, temperature: float = 0.0, batch_size: int = BATCH_SIZE):
        self.llm = llm  # anything with get_completion(prompt, temperature) -> str
        self.temperature = temperature
        self.batch_size = max(1, batch_size)

    def extract(self, text: str) -> List[dict]:
        """Facts in `text`: dicts with entity, attribute, value, valid_from, valid_until."""
        if not text or not text.strip():
            return []
        prompt = FACT_EXTRACTION_PROMPT.format(text=text.strip())
        return parse_facts(self.llm.get_completion(prompt, self.temperature))

    def extract_batch(self, statements: List[str]) -> List[dict]:
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
            prompt = FACT_BATCH_EXTRACTION_PROMPT.format(statements=numbered)
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
