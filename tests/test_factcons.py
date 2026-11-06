import json

import pytest

from harness.factcons_answer import (ANSWER_PROMPT, answer_question, clean_prediction,
                                     normalize_answer, score, substring_match)
from harness.run_factcons import chunk_facts, load_samples, run, run_sample
from tsom.contracts import RunConfig
from tsom.llm import LLM
from tsom.memory import MemorySystem

CONTEXT = ("Here is a list of facts:\n0. Alice is in Lahore.\n1. Bob is in Rome.\n"
           "2. Carol is in Bern.\n3. Alice is in Karachi.\n4. Dan is in Oslo.\n5. Bob is in Paris.\n")


def row(source, context=CONTEXT):
    return {"context": context, "questions": ["Where is Alice?", "Where is Bob?", "Where is Dan?"],
            "answers": [["Karachi"], ["Paris", "France"], "Oslo"], "metadata": {"source": source}}


ROWS = [row("factconsolidation_mh_32k"), row("factconsolidation_sh_32k"),
        row("factconsolidation_sh_6k"), row("factconsolidation_mh_6k"), row("other_thing")]


def cfg(system="amem", tmp=None, **kw):
    return RunConfig(run_name=f"fc_{system}", system=system, backend="fake", model="fake",
                     output_dir=str(tmp) if tmp else "results", **kw)


def test_load_selects_variant_and_size():
    got = load_samples("sh", "6k", loader=lambda: ROWS)
    assert [s["tag"] for s in got] == ["sh_6k"]
    got = load_samples("mh", "all", loader=lambda: ROWS)
    assert [s["tag"] for s in got] == ["mh_6k", "mh_32k"]  # ordered by size
    s = got[0]
    assert s["facts"][0] == (0, "Alice is in Lahore.") and len(s["facts"]) == 6
    assert s["answers"] == [["Karachi"], ["Paris", "France"], ["Oslo"]]  # str -> list
    with pytest.raises(ValueError):
        load_samples("sh", "64k", loader=lambda: ROWS)


def test_chunking_keeps_numbering():
    facts = [(i, f"fact {i}.") for i in range(7)]
    chunks = chunk_facts(facts, 3)
    assert [(a, b) for a, b, _ in chunks] == [(0, 2), (3, 5), (6, 6)]
    assert chunks[1][2] == "3. fact 3.\n4. fact 4.\n5. fact 5."
    assert len(chunk_facts(facts, 0)) == 7  # n < 1 is treated as 1


def test_scoring():
    assert substring_match("The city of Karachi", ["karachi"])
    assert substring_match("Karachi", ["The Karachi Region"])  # gold contains prediction
    assert substring_match("France.", ["Paris", "France"])
    assert not substring_match("", ["Paris"])  # empty prediction never matches
    assert not substring_match("Rome", ["Paris"])
    assert not substring_match("Rome", [""])
    assert normalize_answer("The U.S.A.!") == "usa"
    assert clean_prediction("\n Answer: Paris \nbecause...") == "Paris"
    sc = score("Paris", ["France", "Paris"])
    assert sc["correct"] and sc["f1"] == 1.0


class Scripted(LLM):
    def __init__(self, reply):
        super().__init__("fake", "fake")
        self.reply, self.prompts = reply, []

    def get_completion(self, prompt, temperature=0.7):
        if prompt.startswith("Answer the question using the memory below."):
            self.prompts.append(prompt)
            self.usage.calls += 1
            return self.reply
        return super().get_completion(prompt, temperature)


@pytest.mark.parametrize("system", ["amem", "tsom"])
def test_same_prompt_and_answer_op(system):
    llm = Scripted("Karachi")
    mem = MemorySystem(cfg(system), llm, sample_id="t")
    mem.write("0. Alice is in Lahore.\n3. Alice is in Karachi.", "facts 0-3")
    r = answer_question(mem, llm, "Where is Alice?", 5)
    assert r["prediction"] == "Karachi"
    prompt = llm.prompts[0]
    assert prompt.startswith(ANSWER_PROMPT.split("{context}")[0])  # identical template
    assert prompt.endswith("Question: Where is Alice?\nAnswer:")
    assert "Alice is in Karachi" in prompt
    assert mem.store.ops()[-1].op_type == "answer"
    if system == "tsom":  # facts block carries CURRENT, the instruction's other marker
        assert "[CURRENT] Alice | is | in Karachi (step 3)" in prompt
    else:
        assert "[CURRENT]" not in prompt


def test_tsom_uses_serials_as_observed_at(tmp_path):
    llm = Scripted("x")
    sample = load_samples("sh", "6k", loader=lambda: ROWS)[0]
    run_sample(cfg("tsom", tmp_path, max_questions=1), llm, sample, 3, tmp_path)
    import sqlite3
    db = sqlite3.connect(tmp_path / "memory_sample_sh_6k.sqlite")
    got = db.execute("SELECT entity, observed_at, status FROM facts ORDER BY observed_at").fetchall()
    assert got == [("Alice", 0, "superseded"), ("Bob", 1, "superseded"), ("Carol", 2, "active"),
                   ("Alice", 3, "active"), ("Dan", 4, "active"), ("Bob", 5, "active")]


@pytest.mark.parametrize("system", ["amem", "tsom", "rag"])
def test_run_end_to_end_and_resume(system, tmp_path):
    config = cfg(system, tmp_path, max_samples=1, max_questions=2)
    summary = run(config, "sh", "6k", chunk=3, samples=load_samples("sh", "6k", loader=lambda: ROWS))
    out = tmp_path / f"fc_{system}"
    assert summary["questions"] == 2 and summary["samples"] == 1
    assert summary["accuracy"] == 0.0 and 0.0 <= summary["f1"] <= 1.0  # fake model: "fake answer"
    assert summary["cost"]["answer_llm_calls"] == 2
    assert summary["cost"]["ingest_llm_calls"] == (0 if system == "rag" else summary["cost"]["ingest_llm_calls"])
    if system == "tsom":
        assert summary["tsom"]["fact_ops"].get("ADD", 0) > 0 and summary["tsom"]["failures"] == 0
        assert summary["cost"]["temporal_writes"] == 2  # 6 facts / chunk 3
    else:
        assert summary["tsom"] is None
    for name in ("sample_sh_6k.jsonl", "ops_sample_sh_6k.jsonl", "summary_factcons_sh_6k.json",
                 "summary_factcons_sh_6k.md", "memory_sample_sh_6k.sqlite"):
        assert (out / name).exists(), name
    first_row = json.loads((out / "sample_sh_6k.jsonl").read_text().splitlines()[0])
    assert {"question", "gold", "prediction", "correct", "f1"} <= set(first_row)
    again = run(config, "sh", "6k", chunk=3, samples=load_samples("sh", "6k", loader=lambda: ROWS))
    assert again == summary  # finished sample skipped; summary rebuilt from disk
