"""Structured contracts shared by every component.

Every part of the system (memory, tools, agent, harness) reads and writes
only these objects. `python -m tsom.contracts` prints their JSON schemas.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Dict, List, Literal, Optional

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
    """One append-only fact extracted from a note (temporal layer).

    Rows are never deleted or overwritten; only status, valid_until,
    superseded_by and observation_count may change after insertion.
    """

    fact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sample_id: str
    note_id: str  # A-Mem note id the fact came from
    entity: str
    attribute: str
    value: str
    observed_at: int  # ingestion order (FactConsolidation: the fact's serial number)
    observed_time: Optional[str] = None  # raw session/date string, if any
    valid_from: Optional[str] = None  # only set when the text states it
    valid_until: Optional[str] = None  # only set when the text states it
    status: Literal["active", "superseded", "retracted"] = "active"
    operation: Literal["ADD", "REPLACE", "CORRECT", "REAFFIRM"] = "ADD"
    supersedes: List[str] = []  # fact_ids this fact replaces
    superseded_by: Optional[str] = None
    observation_count: int = 1
    # Tense of the wording only (never changes valid_from/valid_until).
    temporal_status: Literal["PAST", "PRESENT", "UNKNOWN"] = "UNKNOWN"
    # The extractor's own attribute name when it was snapped to an existing one.
    raw_attribute: Optional[str] = None


class MemoryOp(BaseModel):
    """One logged memory operation. Used for latency and cost tables."""

    op_type: Literal["write", "search", "answer", "temporal_write"]
    system: str
    sample_id: Optional[str] = None
    ids_touched: List[str] = []
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    evolved: Optional[bool] = None  # write ops only: did A-Mem evolution change memory
    op_counts: Dict[str, int] = {}  # temporal_write only: fact operations (ADD, REPLACE, ...)
    failures: int = 0  # temporal_write only: extraction/classify/apply steps that failed and were skipped
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
    # tsom only: deterministic "several values at once" rules in the update classifier.
    # Both are off unless a config turns them on; the learned rule has no flag.
    seed_multivalued: bool = False  # attribute is in multivalued_attributes -> ADD
    additive_cue: bool = False  # "also"/"as well"/"too" in the sentence -> ADD
    multivalued_attributes: List[str] = ["speaks", "languages", "children", "siblings",
                                         "hobbies", "member_of"]
    # tsom only: "rules_llm" = deterministic rules first, LLM for what they leave open;
    # "llm_only" = the LLM decides every differing value (only an empty key -> ADD and
    # an identical value -> REAFFIRM are settled without it);
    # "newest_wins" = no classifier: every differing value replaces what is stored, no LLM call.
    classifier_mode: Literal["rules_llm", "llm_only", "newest_wins"] = "rules_llm"
    temperature_c5: float = 0.5  # A-Mem's default for adversarial questions
    # If set, every LLM call uses this, including A-Mem's. None keeps each call's own value.
    temperature: Optional[float] = None
    seed: int = 42
    send_seed: bool = False  # also send seed to the model, not just to the answer-order shuffle
    # tsom only: the agent chooses at question time whether to call get_fact_history.
    use_fact_history: bool = False
    dataset: str = "data/locomo10.json"
    max_samples: Optional[int] = None  # None = all 10 conversations
    # LoCoMo only: the conversations to run, by index (0-9). None = the first max_samples.
    conversations: Optional[List[int]] = None
    max_sessions: Optional[int] = None  # None = every session
    max_questions: Optional[int] = None  # per conversation; None = all
    output_dir: str = "results"

    @model_validator(mode="after")
    def _fact_history_needs_facts(self):
        if self.use_fact_history and self.system != "tsom":
            raise ValueError("use_fact_history needs system='tsom'")
        return self


if __name__ == "__main__":
    for model in (Note, Fact, MemoryOp, RunConfig):
        print(json.dumps(model.model_json_schema(), indent=2))
