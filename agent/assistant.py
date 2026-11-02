"""The assistant agent: one loop that stores turns and answers questions.

Answering follows A-Mem's LoCoMo code (third_party/amem/test_advanced_robust.py)
prompt for prompt, so our baseline numbers are comparable with theirs. The
only difference is that memory is reached through our tools.
"""

from __future__ import annotations

import random
import time
from typing import Optional

from tsom.amem_bridge import parse_keywords_response, parse_plain_text_answer
from tsom.contracts import MemoryOp
from tsom.llm import LLM
from tsom.tools import MemoryTools


class Assistant:
    def __init__(self, tools: MemoryTools, llm: LLM, retrieve_k: int = 10,
                 temperature_c5: float = 0.5, seed: int = 42):
        self.tools = tools
        self.llm = llm
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5
        self.rng = random.Random(seed)

    def observe(self, speaker: str, text: str, time_str: str) -> str:
        # Same turn format A-Mem stores.
        turn = "Speaker " + speaker + "says : " + text
        return self.tools.call("write_memory", turn=turn, time=time_str, speaker=speaker)["note_id"]

    def answer(self, question: str, category: int, gold: str,
               temperature_override: Optional[float] = None) -> dict:
        before = self.llm.usage.snapshot()
        start = time.perf_counter()
        keywords = self._query_keywords(question)
        context = self.tools.call("search_memory", query=keywords, k=self.retrieve_k,
                                  question=question)["notes"]
        prompt, temperature = self._answer_prompt(question, category, gold, context)
        if temperature_override is not None:
            temperature = temperature_override
        try:
            raw = self.llm.get_completion(prompt, temperature=temperature)
        except Exception:
            raw = ""
        used = self.llm.usage.since(before)
        self.tools.memory.store.log_op(MemoryOp(
            op_type="answer", system=self.tools.memory.config.system,
            sample_id=self.tools.memory.sample_id, llm_calls=used.calls,
            prompt_tokens=used.prompt_tokens, completion_tokens=used.completion_tokens,
            latency_s=time.perf_counter() - start,
        ))
        return {"prediction": parse_plain_text_answer(raw), "context": context,
                "prompt_tokens": used.prompt_tokens, "completion_tokens": used.completion_tokens}

    def _query_keywords(self, question: str) -> str:
        prompt = f"""Given the following question, generate several keywords separated by commas.

Question: {question}

Keywords:"""
        return parse_keywords_response(self.llm.get_completion(prompt))

    def _answer_prompt(self, question: str, category: int, gold: str, context: str):
        if category == 5:
            options = ["Not mentioned in the conversation", gold]
            if self.rng.random() >= 0.5:
                options.reverse()
            prompt = f"""Based on the context: {context}, answer the following question. {question}

Select the correct answer: {options[0]} or {options[1]}  Short answer:"""
            return prompt, self.temperature_c5
        if category == 2:
            prompt = f"""Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.

Question: {question} Short answer:"""
            return prompt, 0.7
        prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""
        return prompt, 0.7
