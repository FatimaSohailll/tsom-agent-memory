"""The assistant agent: one loop that stores turns and answers questions.

Answering follows A-Mem's LoCoMo code (third_party/amem/test_advanced_robust.py)
prompt for prompt, so our baseline numbers are comparable with theirs. The
only difference is that memory is reached through our tools.
"""

from __future__ import annotations

import random
import re
import time

from tsom.amem_bridge import parse_keywords_response, parse_plain_text_answer
from tsom.contracts import MemoryOp
from tsom.llm import LLM
from tsom.tools import MemoryTools

TOOL_PROMPT = """Choose the memory tool for this question.
search_memory: what is true now, or anything else.
get_fact_history: a past value, or how something changed over time.

Question: {question}

Reply in this format. Add the ENTITY line only for get_fact_history, with the person or thing it is about.
TOOL: search_memory or get_fact_history
ENTITY: the name"""
_HISTORY = re.compile(r"TOOL:\s*get_fact_history", re.IGNORECASE)
_ENTITY = re.compile(r"ENTITY:\s*(.+)", re.IGNORECASE)


class Assistant:
    def __init__(self, tools: MemoryTools, llm: LLM, retrieve_k: int = 10,
                 temperature_c5: float = 0.5, seed: int = 42, use_fact_history: bool = False):
        self.tools = tools
        self.llm = llm
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5
        self.rng = random.Random(seed)
        self.use_fact_history = use_fact_history

    def observe(self, speaker: str, text: str, time_str: str) -> str:
        # Same turn format A-Mem stores.
        turn = "Speaker " + speaker + "says : " + text
        return self.tools.call("write_memory", turn=turn, time=time_str)["note_id"]

    def answer(self, question: str, category: int, gold: str) -> dict:
        before = self.llm.usage.snapshot()
        start = time.perf_counter()
        keywords = self._query_keywords(question)
        context = self.tools.call("search_memory", query=keywords, k=self.retrieve_k)["notes"]
        tools = [{"tool": "search_memory", "query": keywords, "k": self.retrieve_k}]
        if self.use_fact_history:
            entity, history = self._chosen_history(question)
            if entity is not None:
                tools.append({"tool": "get_fact_history", "entity": entity, "found": bool(history)})
            if history:
                context = f"Fact history:\n{history}\n\n{context}"
        prompt, temperature = self._answer_prompt(question, category, gold, context)
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
        return {"prediction": parse_plain_text_answer(raw), "context": context, "tools": tools,
                "prompt_tokens": used.prompt_tokens, "completion_tokens": used.completion_tokens}

    def _chosen_history(self, question: str):
        """(entity, history) the model asks for; (None, "") when it chooses plain search
        or its reply cannot be used."""
        reply = self.llm.get_completion(TOOL_PROMPT.format(question=question), temperature=0.0)
        entity = _ENTITY.search(reply)
        if not _HISTORY.search(reply) or not entity:
            return None, ""
        name = entity.group(1).strip()
        return name, self.tools.call("get_fact_history", entity=name)["facts"]

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
