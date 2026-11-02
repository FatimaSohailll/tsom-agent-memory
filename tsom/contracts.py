"""Structured contracts shared by every component.

Every part of the system (memory, tools, agent, harness) reads and writes
only these objects. `python -m tsom.contracts` prints their JSON schemas.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class Note(BaseModel):
    """An A-Mem note. Field names follow A-Mem's MemoryNote."""

    id: str = Field(default_factory=new_id)
    content: str
    timestamp: str
    keywords: List[str] = []
    tags: List[str] = []
    context: str = "General"
    links: List[str] = []  # ids of linked notes


class Fact(BaseModel):
    """A typed fact with a validity interval (our addition, Gap 3)."""

    id: str = Field(default_factory=new_id)
    subject: str
    predicate: str
    object: str
    valid_from: str
    valid_to: Optional[str] = None  # None while the fact is current
    supersedes: Optional[str] = None  # id of the fact this one replaced
    source_note_id: Optional[str] = None
    status: Literal["active", "retired", "rejected"] = "active"
    conflict_with: Optional[str] = None  # id of an active fact this one contradicts (Gap 1)


Label = Literal["additive", "redundant", "corrective", "contradictory"]


class FactOp(BaseModel):
    """One fact-layer decision for a turn (Gap 1)."""

    predicate: str
    value: str
    raw_label: str  # the label the model gave
    label: Label  # the label that was applied
    fact_id: Optional[str] = None  # new fact stored, if any
    retired_id: Optional[str] = None  # fact retired by this decision, if any
    conflict_with: Optional[str] = None


class MemoryOp(BaseModel):
    """One logged memory operation. Used for latency and cost tables."""

    op_type: Literal["write", "search", "fact_history", "inspect", "answer"]
    system: str
    sample_id: Optional[str] = None
    ids_touched: List[str] = []
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    evolved: Optional[bool] = None  # write ops only: did A-Mem evolution change memory
    labels: List[str] = []  # write ops only: applied fact labels (tsom)
    raw_labels: List[str] = []  # write ops only: labels as the model gave them (tsom)
    parse_failures: int = 0  # write ops only: extraction outputs that could not be parsed
    session: Optional[int] = None  # write ops only: LoCoMo session being replayed
    gate: Optional[str] = None  # write ops only: gate reason (Gap 2); None when ungated
    neighbour_similarity: Optional[float] = None  # write ops only: max similarity the gate saw
    timestamp: float = Field(default_factory=time.time)


class RunConfig(BaseModel):
    """Everything needed to reproduce one run."""

    run_name: str
    system: Literal["amem", "rag", "tsom"] = "amem"
    backend: Literal["ollama", "openai", "vllm", "fake"] = "ollama"
    model: str = "qwen2.5:3b"
    base_url: Optional[str] = None  # vLLM or another OpenAI-compatible server
    embedder: str = "all-MiniLM-L6-v2"
    retrieve_k: int = 10
    temperature_c5: float = 0.5  # A-Mem's default for adversarial questions
    tau: float = 0.8  # gate similarity threshold (Phase 2)
    gate_link_k: int = 3  # links a skipped note gets to its most similar notes (Gap 2)
    seed: int = 42
    use_classifier: bool = False  # Gap 1 (Phase 2)
    use_gate: bool = False  # Gap 2 (Phase 2)
    use_validity: bool = False  # Gap 3 (Phase 2)
    dataset: str = "data/locomo10.json"
    max_samples: Optional[int] = None  # None = all 10 conversations
    conversation_ids: Optional[List[int]] = None  # explicit LoCoMo indices; overrides max_samples
    max_sessions: Optional[int] = None  # None = every session
    max_questions: Optional[int] = None  # per conversation; None = all
    output_dir: str = "results"

    @model_validator(mode="after")
    def _baselines_have_no_fact_layer(self):
        if self.system in ("amem", "rag") and (self.use_classifier or self.use_validity):
            raise ValueError(f'use_classifier and use_validity need system="tsom"; '
                             f'system="{self.system}" is a baseline and has no fact layer')
        if self.use_gate and self.system != "tsom":
            raise ValueError(f'use_gate needs system="tsom"; the gate reads the fact labels, '
                             f'and system="{self.system}" has no fact layer')
        return self


if __name__ == "__main__":
    for model in (Note, Fact, FactOp, MemoryOp, RunConfig):
        print(json.dumps(model.model_json_schema(), indent=2))
