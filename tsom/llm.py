"""LLM provider setup.

One small wrapper for every backend. It has the same `get_completion(prompt,
temperature)` method as A-Mem's controllers, so A-Mem's code can use it
unchanged, and it counts calls, tokens and seconds for the cost tables.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass

# Same system message and output cap as A-Mem's robust controllers.
SYSTEM_MESSAGE = "Follow the format specified in the prompt exactly. Do not add extra commentary."
MAX_TOKENS = 1000


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0

    def snapshot(self) -> "Usage":
        return Usage(self.calls, self.prompt_tokens, self.completion_tokens, self.seconds)

    def since(self, before: "Usage") -> "Usage":
        return Usage(self.calls - before.calls,
                     self.prompt_tokens - before.prompt_tokens,
                     self.completion_tokens - before.completion_tokens,
                     self.seconds - before.seconds)


class LLM:
    def __init__(self, backend: str, model: str, base_url: str | None = None, retries: int = 2):
        self.backend = backend
        self.model = model
        self.retries = retries
        self.usage = Usage()
        if backend == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif backend == "vllm":
            from openai import OpenAI
            self.client = OpenAI(base_url=base_url or "http://localhost:30000/v1",
                                 api_key=os.getenv("OPENAI_API_KEY", "EMPTY"))
        elif backend == "ollama":
            import ollama
            self.client = ollama.Client(host=base_url) if base_url else ollama.Client()
        elif backend == "fake":
            self.client = None
        else:
            raise ValueError(f"unknown backend {backend}")

    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        start = time.perf_counter()
        for attempt in range(self.retries + 1):
            try:
                text, p_tok, c_tok = self._call(prompt, temperature)
                break
            except Exception:
                if attempt == self.retries:
                    raise
                time.sleep(2 ** attempt)
        self.usage.calls += 1
        self.usage.prompt_tokens += p_tok
        self.usage.completion_tokens += c_tok
        self.usage.seconds += time.perf_counter() - start
        return text

    def check(self):
        """Fail early with a clear message if the backend is not reachable."""
        try:
            self.get_completion("Reply with exactly one word: READY", temperature=0.0)
        except Exception as e:
            raise ConnectionError(f"Cannot reach {self.backend} model {self.model}: {e}") from e

    def _call(self, prompt: str, temperature: float):
        messages = [{"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt}]
        if self.backend in ("openai", "vllm"):
            r = self.client.chat.completions.create(model=self.model, messages=messages,
                                                    temperature=temperature, max_tokens=MAX_TOKENS)
            u = r.usage
            return r.choices[0].message.content or "", u.prompt_tokens if u else 0, u.completion_tokens if u else 0
        if self.backend == "ollama":
            r = self.client.chat(model=self.model, messages=messages,
                                 options={"temperature": temperature, "num_predict": MAX_TOKENS})
            return r["message"]["content"], r.get("prompt_eval_count") or 0, r.get("eval_count") or 0
        return fake_completion(prompt), len(prompt.split()), 5


def fake_completion(prompt: str) -> str:
    """Deterministic stand-in for an LLM, used by tests and offline smoke runs.

    It returns text in the shapes A-Mem's parsers expect, so the whole pipeline
    runs without a model. Its answers are meaningless; only the plumbing is tested.
    """
    if prompt.startswith("A memory already contains this fact:"):
        new_part = prompt.split("A new fact arrives:")[-1].split("Classify how")[0].lower()
        if any(w in new_part for w in ("actually", "mistaken", "correction")):
            return "OPERATION: CORRECT\nREASON: fake"
        if " also " in new_part:
            return "OPERATION: ADD\nREASON: fake"
        return "OPERATION: REPLACE\nREASON: fake"
    if prompt.startswith("Extract the atomic facts stated in each numbered statement."):
        return _fake_facts(prompt.split("\nStatements:\n")[-1], batch=True)
    if prompt.startswith("Extract the atomic facts stated in the text."):
        return _fake_facts(prompt.split("\nText:\n")[-1], batch=False)
    h = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
    if prompt.startswith("You are an AI memory evolution agent"):
        decision = ["NO_EVOLUTION", "STRENGTHEN", "UPDATE_NEIGHBOR", "STRENGTHEN_AND_UPDATE"][h % 4]
        return f"DECISION: {decision}\nREASON: fake"
    if "CONNECTIONS:" in prompt:
        return "CONNECTIONS: 0\nTAGS: chat, fake, linked"
    if "NEIGHBOR 0:" in prompt:
        n = int(prompt.split("(continue for all ")[1].split(" ")[0])
        return "\n\n".join(f"NEIGHBOR {i}:\nCONTEXT: updated by fake\nTAGS: chat, fake" for i in range(n))
    if prompt.startswith("Analyze the following content"):
        words = [w.strip(".,!?:").lower() for w in prompt.split("Content for analysis:")[-1].split() if len(w) > 4][:5]
        words = words or ["chat", "turn", "talk"]
        return f"KEYWORDS: {', '.join(words)}\nCONTEXT: A conversation turn.\nTAGS: chat, fake, dialogue"
    if prompt.startswith("Given the following question, generate several keywords"):
        return prompt.split("Question:")[-1].split("\n")[0].strip()
    if "Short answer:" in prompt:
        return "answer " + format(h % 10**6, "06d")
    return "READY"


def _fake_facts(body: str, batch: bool) -> str:
    """Fake fact extraction: each "X is Y." sentence becomes (X, is, Y)."""
    blocks = []
    for n, line in enumerate(body.strip().splitlines(), 1):
        source = n
        if batch:
            m = re.match(r"\s*(\d+)\.\s+(.*)", line)
            if not m:
                continue
            source, line = int(m.group(1)), m.group(2)
        for sentence in line.split("."):
            if " is " not in sentence:
                continue
            entity, value = sentence.split(" is ", 1)
            lines = [f"FACT {len(blocks) + 1}:"]
            if batch:
                lines.append(f"SOURCE: {source}")
            lines += [f"ENTITY: {entity.strip()}", "ATTRIBUTE: is", f"VALUE: {value.strip()}",
                      "VALID_FROM: NONE", "VALID_UNTIL: NONE"]
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "NO_FACTS"
