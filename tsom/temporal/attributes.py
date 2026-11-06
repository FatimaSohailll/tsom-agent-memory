"""Keep attribute names stable across writes ("attribute drift").

Facts are keyed by (entity, attribute). If the extractor writes `speaks` once and
`speaks_language` later, the two never meet and no update is ever classified.
Two defences:

1. known_attributes(): names already in use, offered to the extractor so it can
   reuse them.
2. canonicalize(): after extraction, a new name is snapped to a name the same
   entity already has, but only when that is safe:
   - same canonical form (equal once filler words are dropped), or
   - embedding cosine at or above a high threshold.
   Sibling relations score high with embeddings (`speaks` ~ `speaks_to` 0.82,
   `head_of_state` ~ `head_of_government` 0.77 with MiniLM), so the threshold
   is deliberately strict; borderline pairs are left apart and the prompt hint
   has to handle them. A wrong merge makes a REPLACE supersede a true fact,
   which is worse than the drift.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .normalize import norm

log = logging.getLogger("tsom.attributes")

# Dropped before comparing. Prepositions other than "of" stay: works_at / works_for
# and speaks / speaks_to are different relations.
FILLER = {"of", "the", "a", "an", "language", "languages", "name"}


def _stem(token: str) -> str:
    return token[:-1] if len(token) > 3 and token.endswith("s") and not token.endswith("ss") else token


def canonical_form(attribute: str) -> Tuple[str, ...]:
    """Sorted stem tokens without filler: speaks_language -> ("speak",)."""
    tokens = [_stem(t) for t in re.split(r"[^a-z0-9]+", norm(attribute)) if t]
    kept = sorted({t for t in tokens if t not in FILLER})
    return tuple(kept) if kept else tuple(sorted(set(tokens)))


class AttributeCanonicalizer:
    def __init__(self, store, encode: Optional[Callable[[List[str]], np.ndarray]] = None,
                 threshold: float = 0.88, hint_limit: int = 20):
        self.store = store
        self.encode = encode  # list[str] -> 2D array; None disables the embedding stage
        self.threshold = threshold
        self.hint_limit = hint_limit
        self.snaps: Counter = Counter()  # "canonical" / "embedding"
        self._vectors: Dict[str, np.ndarray] = {}

    # ---- 1. names offered to the extractor ----

    def known_attributes(self, sample_id: str, text: str) -> List[str]:
        """Attribute names of entities mentioned in `text` first, then the most
        used names in the sample, up to hint_limit."""
        names: List[str] = []
        seen = set()

        def add(attribute: str):
            if len(names) < self.hint_limit and norm(attribute) not in seen:
                seen.add(norm(attribute))
                names.append(attribute)

        lowered = text.lower()
        for entity in self.store.entity_keys(sample_id):
            if len(entity) >= 2 and entity in lowered and re.search(
                    r"(?<!\w)" + re.escape(entity) + r"(?!\w)", lowered):
                for attribute, _ in self.store.attribute_counts(sample_id, entity):
                    add(attribute)
        for attribute, _ in self.store.attribute_counts(sample_id):
            add(attribute)
        return names

    # ---- 2. snapping after extraction ----

    def canonicalize(self, sample_id: str, entity: str, attribute: str) -> Tuple[str, Optional[str]]:
        """Return (attribute to use, reason it was changed or None)."""
        existing = [a for a, _ in self.store.attribute_counts(sample_id, entity)]
        if not existing or norm(attribute) in {norm(a) for a in existing}:
            return attribute, None
        form = canonical_form(attribute)
        for other in existing:
            if canonical_form(other) == form:
                self.snaps["canonical"] += 1
                return other, "same canonical form"
        if self.encode is not None:
            best, sim = max(((o, self._similarity(attribute, o)) for o in existing), key=lambda p: p[1])
            if sim >= self.threshold:
                self.snaps["embedding"] += 1
                return best, f"embedding similarity {sim:.2f}"
        return attribute, None

    def _vector(self, attribute: str) -> np.ndarray:
        if attribute not in self._vectors:
            v = np.asarray(self.encode([attribute.replace("_", " ")])[0], dtype=float)
            self._vectors[attribute] = v / (np.linalg.norm(v) or 1.0)
        return self._vectors[attribute]

    def _similarity(self, a: str, b: str) -> float:
        return float(self._vector(a) @ self._vector(b))
