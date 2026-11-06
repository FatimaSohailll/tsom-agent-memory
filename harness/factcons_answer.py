"""Answering and scoring for FactConsolidation (MemoryAgentBench).

Kept apart from agent/assistant.py, which is LoCoMo-specific. The same prompt
and retrieval call are used for every memory system, so the comparison is fair.
"""

from __future__ import annotations

import logging
import re
import string
import time
from typing import List

from harness.metrics import f1
from tsom.contracts import MemoryOp

log = logging.getLogger("harness")

ANSWER_PROMPT = """Answer the question using the memory below. Facts are numbered; when facts
conflict, the one with the larger number (or marked CURRENT) is the latest
and must be used. Answer with only the entity name, no explanation.
Memory:
{context}
Question: {question}
Answer:"""


def answer_question(memory, llm, question: str, k: int, temperature: float = 0.0) -> dict:
    """Retrieve with memory.search(question, k), ask the model, log an "answer" op."""
    before = llm.usage.snapshot()
    start = time.perf_counter()
    context = memory.search(question, k)["notes"]
    prompt = ANSWER_PROMPT.format(context=context, question=question)
    try:
        raw = llm.get_completion(prompt, temperature=temperature)
    except Exception as e:  # same convention as the LoCoMo agent: score it as a miss
        log.warning("answer call failed for %r: %s", question, e)
        raw = ""
    used = llm.usage.since(before)
    memory.store.log_op(MemoryOp(
        op_type="answer", system=memory.config.system, sample_id=memory.sample_id,
        llm_calls=used.calls, prompt_tokens=used.prompt_tokens,
        completion_tokens=used.completion_tokens, latency_s=time.perf_counter() - start,
    ))
    return {"prediction": clean_prediction(raw), "context": context,
            "prompt_tokens": used.prompt_tokens, "completion_tokens": used.completion_tokens}


def clean_prediction(raw: str) -> str:
    """First non-empty line of the reply, with an optional "Answer:" prefix removed."""
    for line in (raw or "").splitlines():
        line = line.strip()
        if line:
            return re.sub(r"^answer\s*:\s*", "", line, flags=re.IGNORECASE).strip()
    return ""


def normalize_answer(s: str) -> str:
    """Lowercase, drop punctuation and articles, collapse whitespace (SQuAD style)."""
    s = "".join(ch for ch in str(s).lower() if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def substring_match(prediction: str, golds: List[str]) -> bool:
    """Normalized prediction contains a gold answer, or a gold answer contains it."""
    pred = normalize_answer(prediction)
    if not pred:
        return False
    for gold in golds:
        g = normalize_answer(gold)
        if g and (g in pred or pred in g):
            return True
    return False


def score(prediction: str, golds: List[str]) -> dict:
    return {"correct": substring_match(prediction, golds),
            "f1": max((f1(prediction, g) for g in golds), default=0.0)}
