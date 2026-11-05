import json

import pytest

from harness.factcons_answer import (ANSWER_PROMPT, answer_question, clean_prediction,
                                     mark_stale, normalize_answer, older_values, score,
                                     substring_match)
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


CAPITAL = [(0, "The capital of France is Rome."), (1, "Bob was born in the city of Lyon."),
           (2, "The capital of France is Paris."), (3, "Bob was born in the city of Paris.")]


def test_older_values_are_earlier_values_of_the_same_fact():
    question = "What is the capital of France?"
    assert older_values(CAPITAL, question, ["Paris"]) == ["Rome"]
    assert older_values(CAPITAL, "Where was Bob born?", ["Paris"]) == ["Lyon"]


def test_multi_hop_keeps_every_fact_ending_in_the_answer():
    found = older_values(CAPITAL, "Where was the spouse of Ann born?", ["Paris"], closest=False)
    assert found == ["Lyon", "Rome"]


def test_a_fact_that_never_changed_has_no_older_value():
    assert older_values([(0, "Dan is in Oslo.")], "Where is Dan?", ["Oslo"]) == []


def test_an_answer_that_is_not_the_newest_value_has_no_older_value():
    assert older_values(CAPITAL, "What is the capital of France?", ["Rome"]) == []


def test_older_values_match_whole_words():
    facts = [(0, "The home of Al is Oslo."), (1, "Albania is in Europe.")]
    assert older_values(facts, "Where is Al?", ["Al"]) == []


@pytest.mark.parametrize("prediction,stale", [("Rome", True), ("rome.", True), ("Paris", False),
                                              ("Madrid", False), ("", False)])
def test_only_a_wrong_answer_with_an_older_value_is_stale(prediction, stale):
    row = {"question": "What is the capital of France?", "gold": ["Paris"], "prediction": prediction,
           **score(prediction, ["Paris"])}
    marked = mark_stale(row, {"tag": "sh_6k", "facts": CAPITAL})
    assert marked["older"] == ["Rome"] and marked["stale"] is stale


def test_summary_counts_stale_answers_and_keeps_them_on_resume(tmp_path, monkeypatch):
    replies = iter(["Lahore", "Paris", "Oslo"])
    monkeypatch.setattr("harness.run_factcons.answer_question", lambda *a, **k: {
        "prediction": next(replies), "context": "", "prompt_tokens": 0, "completion_tokens": 0})
    config = cfg("rag", tmp_path, max_samples=1)
    samples = load_samples("sh", "6k", loader=lambda: ROWS)
    summary = run(config, "sh", "6k", chunk=3, samples=samples)
    assert summary["stale"] == {"changed_questions": 2, "stale_answers": 1, "stale_rate": 0.5}
    assert summary["per_sample"]["sh_6k"]["stale_answers"] == 1
    assert "1 of 2" in (tmp_path / "fc_rag" / "summary_factcons_sh_6k.md").read_text()
    assert run(config, "sh", "6k", chunk=3, samples=samples) == summary


def test_rows_saved_before_stale_scoring_are_marked_on_resume(tmp_path, monkeypatch):
    monkeypatch.setattr("harness.run_factcons.answer_question", lambda *a, **k: {
        "prediction": "Lahore", "context": "", "prompt_tokens": 0, "completion_tokens": 0})
    config = cfg("rag", tmp_path, max_samples=1, max_questions=1)
    samples = load_samples("sh", "6k", loader=lambda: ROWS)
    run(config, "sh", "6k", chunk=3, samples=samples)
    saved = tmp_path / "fc_rag" / "sample_sh_6k.jsonl"
    row = json.loads(saved.read_text())
    saved.write_text(json.dumps({k: v for k, v in row.items() if k not in ("older", "stale")}) + "\n")
    assert run(config, "sh", "6k", chunk=3, samples=samples)["stale"]["stale_answers"] == 1


def test_multi_hop_summary_is_labelled_an_estimate(tmp_path):
    config = cfg("rag", tmp_path, max_samples=1, max_questions=1)
    run(config, "mh", "6k", chunk=3, samples=load_samples("mh", "6k", loader=lambda: ROWS))
    assert "estimate" in (tmp_path / "fc_rag" / "summary_factcons_mh_6k.md").read_text()
