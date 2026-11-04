"""Structured contracts shared by every component.

Every part of the system (memory, tools, agent, harness) reads and writes
only these objects. `python -m tsom.contracts` prints their JSON schemas.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


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


class MemoryOp(BaseModel):
    """One logged memory operation. Used for latency and cost tables."""

    op_type: Literal["write", "search", "answer"]
    system: str
    sample_id: Optional[str] = None
    ids_touched: List[str] = []
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    evolved: Optional[bool] = None  # write ops only: did A-Mem evolution change memory
    timestamp: float = Field(default_factory=time.time)


class RunConfig(BaseModel):
    """Everything needed to reproduce one run."""

    run_name: str
    system: Literal["amem", "rag"] = "amem"
    backend: Literal["ollama", "openai", "vllm", "fake"] = "ollama"
    model: str = "qwen2.5:3b"
    base_url: Optional[str] = None  # vLLM or another OpenAI-compatible server
    embedder: str = "all-MiniLM-L6-v2"
    retrieve_k: int = 10
    temperature_c5: float = 0.5  # A-Mem's default for adversarial questions
    seed: int = 42
    dataset: str = "data/locomo10.json"
    max_samples: Optional[int] = None  # None = all 10 conversations
    max_sessions: Optional[int] = None  # None = every session
    max_questions: Optional[int] = None  # per conversation; None = all
    output_dir: str = "results"


if __name__ == "__main__":
    for model in (Note, MemoryOp, RunConfig):
        print(json.dumps(model.model_json_schema(), indent=2))
