"""The conflict gate (Gap 2): run A-Mem's evolution only when something changed.

A-Mem runs its evolution calls (decision, strengthen, update neighbours) on
every write. The gate runs them only if this turn's facts were labelled
corrective or contradictory, or a stored note is at least tau similar to the
new note. Otherwise the note is linked to its most similar stored notes by
embedding alone, with no LLM call. Note construction (A-Mem's analysis call)
always runs. The gate itself never calls the LLM, so it works the same with
every backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np

from .amem_bridge import RobustAgenticMemorySystem

CONFLICT_LABELS = ("corrective", "contradictory")


@dataclass
class GateDecision:
    run_evolution: bool
    reason: str  # no_neighbours, conflict, neighbour or skipped
    max_similarity: Optional[float]


def decide(labels: Iterable[str], max_similarity: Optional[float], tau: float) -> GateDecision:
    """The gate rule. max_similarity is None when memory is empty."""
    if max_similarity is None:
        return GateDecision(False, "no_neighbours", None)  # A-Mem has nothing to evolve either
    if any(label in CONFLICT_LABELS for label in labels):
        return GateDecision(True, "conflict", max_similarity)
    if max_similarity >= tau:
        return GateDecision(True, "neighbour", max_similarity)
    return GateDecision(False, "skipped", max_similarity)


class GatedAgenticMemory(RobustAgenticMemorySystem):
    """A-Mem with the gate in front of process_memory. Nothing else changes."""

    def __init__(self, *args, tau: float = 0.8, link_k: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau
        self.link_k = link_k
        self.labels: List[str] = []
        self.last_decision: Optional[GateDecision] = None

    def set_write_context(self, labels: Iterable[str]):
        """The applied fact labels of the turn about to be written."""
        self.labels = list(labels)

    def similarities(self, content: str) -> Optional[np.ndarray]:
        """Cosine similarity of content to every stored note, in memory order,
        computed the way A-Mem's retriever ranks neighbours."""
        embeddings = self.retriever.embeddings
        if not self.memories or embeddings is None or len(embeddings) == 0:
            return None
        query = np.asarray(self.retriever.model.encode([content])[0], dtype=float)
        stored = np.asarray(embeddings, dtype=float)
        norms = np.linalg.norm(stored, axis=1) * np.linalg.norm(query)
        return stored @ query / np.where(norms == 0, 1.0, norms)

    def process_memory(self, note) -> tuple:
        sims = self.similarities(note.content)
        decision = decide(self.labels, None if sims is None else float(sims.max()), self.tau)
        self.last_decision = decision
        self.labels = []  # used once
        if decision.run_evolution:
            return super().process_memory(note)
        if sims is not None and self.link_k > 0:
            nearest = np.argsort(sims)[::-1][: self.link_k]
            note.links.extend(int(i) for i in nearest)
        return False, note
