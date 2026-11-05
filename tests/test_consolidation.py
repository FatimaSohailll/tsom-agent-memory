from harness.run_locomo import cost_summary
from tsom.contracts import Fact, RunConfig
from tsom.llm import LLM
from tsom.memory import MemorySystem
from tsom.store import Store
from tsom.temporal.consolidator import Consolidator


class Scripted(LLM):
    """Fake backend with optional scripted fact extraction / evolution decision."""

    def __init__(self, facts=None, decision=None):
        super().__init__("fake", "fake")
        self.facts, self.decision = list(facts or []), decision

    def get_completion(self, prompt, temperature=0.7):
        if self.facts and prompt.startswith("Extract the atomic facts stated in the text."):
            self.usage.calls += 1
            return self.facts.pop(0)
        if self.decision and prompt.startswith("You are an AI memory evolution agent"):
            self.usage.calls += 1
            return f"DECISION: {self.decision}\nREASON: forced"
        return super().get_completion(prompt, temperature)


def fact_text(value, attr="lives_in", entity="Alice"):
    return f"FACT 1:\nENTITY: {entity}\nATTRIBUTE: {attr}\nVALUE: {value}\nVALID_FROM: NONE\nVALID_UNTIL: NONE"


def system(llm, name="tsom"):
    return MemorySystem(RunConfig(run_name="t", system=name, backend="fake", model="fake"),
                        llm, sample_id="s0")


def test_three_replaces_chain():
    mem = system(LLM("fake", "fake"))
    for i, city in enumerate(["Lahore", "Karachi", "Paris"], 1):
        mem.write(f"Alice is in {city}.", f"{i} May 2023")
    hist = mem.store.fact_history("s0", "alice", "is")
    assert [f.value for f in hist] == ["in Lahore", "in Karachi", "in Paris"]
    assert [f.status for f in hist] == ["superseded", "superseded", "active"]
    assert [f.operation for f in hist] == ["ADD", "REPLACE", "REPLACE"]
    a, b, c = hist
    assert a.superseded_by == b.fact_id and b.superseded_by == c.fact_id
    assert b.supersedes == [a.fact_id] and c.supersedes == [b.fact_id]
    assert [f.observed_at for f in hist] == [1, 2, 3]
    assert hist[1].observed_time == "2 May 2023"
    assert [f.fact_id for f in mem.store.active_facts("s0", "Alice", "is")] == [c.fact_id]
    assert mem.classifier.op_counts == {"ADD": 1, "REPLACE": 2}


def test_correct_leaves_retracted_row():
    llm = Scripted([fact_text("Lahore"), fact_text("Karachi")])
    mem = system(llm)
    mem.write("I live in Lahore.", "t1")
    mem.write("Actually, I live in Karachi.", "t2")  # fake classifier: "actually" -> CORRECT
    a, b = mem.store.fact_history("s0", "alice", "lives_in")
    assert (a.status, b.status) == ("retracted", "active")
    assert a.superseded_by == b.fact_id and b.operation == "CORRECT" and b.supersedes == [a.fact_id]
    assert a.valid_until is None


def test_reaffirm_bumps_count_without_insert():
    llm = Scripted([fact_text("Lahore"), fact_text("lahore.")])
    mem = system(llm)
    mem.write("I live in Lahore.", "t1")
    mem.write("Still in Lahore.", "t2")
    (f,) = mem.store.fact_history("s0", "alice", "lives_in")
    assert f.observation_count == 2 and f.status == "active"


def test_neighbour_rewrite_creates_note_version():
    llm = Scripted(decision="UPDATE_NEIGHBOR")
    mem = system(llm)
    first = mem.write("Alice is happy.", "t1")
    old = mem.store.get_note(first)
    second = mem.write("Alice is moving house.", "t2")
    assert mem.amem.memories[first].context == "updated by fake"
    (v,) = mem.store.note_versions(first)
    assert v["version"] == 1 and v["changed_by_note_id"] == second
    assert v["context"] == old.context and v["context"] != "updated by fake"
    assert mem.store.note_versions(second) == []


def test_temporal_op_logged():
    mem = system(LLM("fake", "fake"))
    mem.write("Alice is in Lahore.", "t1")
    mem.write("Alice is in Karachi.", "t2")
    ops = mem.store.ops()
    assert [o.op_type for o in ops] == ["write", "temporal_write"] * 2
    t = ops[-1]
    assert t.llm_calls == 2 and t.op_counts == {"REPLACE": 1}  # extraction + classification
    assert ops[0].llm_calls >= 1


def test_serial_lines_set_observed_at():
    mem = system(LLM("fake", "fake"))
    mem.write("12. Bob is in Rome.\n15. Bob is in Oslo.", "t1")
    hist = mem.store.fact_history("s0", "bob", "is")
    assert [f.observed_at for f in hist] == [12, 15]
    assert [f.status for f in hist] == ["superseded", "active"]
    mem.write("Carol is in Bern.", "t2")
    assert mem.store.fact_history("s0", "carol", "is")[0].observed_at == 16


def test_amem_system_makes_no_facts():
    mem = system(LLM("fake", "fake"), name="amem")
    mem.write("Alice is in Lahore.", "t1")
    mem.write("Alice is in Karachi.", "t2")
    assert mem.store.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 0
    assert mem.store.conn.execute("SELECT COUNT(*) FROM note_versions").fetchone()[0] == 0
    assert [o.op_type for o in mem.store.ops()] == ["write", "write"]
    assert not hasattr(mem, "extractor")


# ---- Consolidator directly ----

def base(**kw):
    return {**dict(sample_id="s0", note_id="n", entity="E", attribute="a", value="v1",
                   observed_at=1), **kw}


def test_replace_valid_until_from_new_valid_from():
    store = Store()
    c = Consolidator(store)
    old = Fact(**base(valid_from="2018"))
    store.add_fact(old)
    new = c.apply("s0", "n2", {"entity": "E", "attribute": "a", "value": "v2",
                               "valid_from": "2020", "valid_until": None}, "REPLACE", [old], 2, None)
    got = store.get_fact(old.fact_id)
    assert got.valid_until == "2020" and got.status == "superseded" and got.superseded_by == new.fact_id
    assert got.value == "v1"  # untouched


def test_replace_without_dates_keeps_null():
    store = Store()
    old = Fact(**base())
    store.add_fact(old)
    Consolidator(store).apply("s0", "n2", {"entity": "E", "attribute": "a", "value": "v2"},
                              "REPLACE", [old], 2, None)
    assert store.get_fact(old.fact_id).valid_until is None


def test_add_historical():
    store = Store()
    old = Fact(**base(value="v2", valid_from="2022-03"))
    store.add_fact(old)
    new = Consolidator(store).apply(
        "s0", "n2", {"entity": "E", "attribute": "a", "value": "v1", "valid_from": "2019"},
        "ADD_HISTORICAL", [old], 2, None)
    assert (new.status, new.valid_until, new.superseded_by) == ("superseded", "2022-03", old.fact_id)
    assert new.valid_from == "2019"
    assert store.get_fact(old.fact_id).status == "active"  # old row untouched
    assert [f.fact_id for f in store.active_facts("s0", "e", "a")] == [old.fact_id]


def test_replace_multiple_old_and_degenerate_cases():
    import pytest
    store = Store()
    a, b = Fact(**base(value="x")), Fact(**base(value="y", observed_at=2))
    store.add_fact(a)
    store.add_fact(b)
    c = Consolidator(store)
    new = c.apply("s0", "n", {"entity": "E", "attribute": "a", "value": "z"}, "REPLACE", [a, b], 3, None)
    assert new.supersedes == [a.fact_id, b.fact_id]
    assert all(store.get_fact(f.fact_id).superseded_by == new.fact_id for f in (a, b))
    assert c.apply("s0", "n", {"entity": "E", "attribute": "q", "value": "1"}, "CORRECT", [], 4, None
                   ).operation == "ADD"
    with pytest.raises(ValueError):
        c.apply("s0", "n", {"entity": "E", "attribute": "a", "value": "1"}, "REAFFIRM", [], 5, None)
    with pytest.raises(ValueError):
        c.apply("s0", "n", {"entity": "E", "attribute": "a", "value": "1"}, "NOPE", [], 5, None)


# ---- failure handling, logging, harness cost ----

import logging

import pytest


class FlakyClassify(Scripted):
    """Raises on the first classification call, like a model server hiccup."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.raised = 0

    def get_completion(self, prompt, temperature=0.7):
        if prompt.startswith("A memory already contains this fact:") and not self.raised:
            self.raised += 1
            raise RuntimeError("model unavailable")
        return super().get_completion(prompt, temperature)


def test_failing_fact_is_skipped_and_run_continues(caplog):
    mem = system(FlakyClassify())
    first = mem.write("Alice is in Lahore.", "t1")
    with caplog.at_level(logging.WARNING, logger="tsom.memory"):
        second = mem.write("Alice is in Karachi.", "t2")  # classify raises -> skipped
        third = mem.write("Alice is in Paris.", "t3")     # run continues
    assert all(mem.store.get_note(n) is not None for n in (first, second, third))  # notes stay
    hist = mem.store.fact_history("s0", "alice", "is")
    assert [(f.value, f.status) for f in hist] == [("in Lahore", "superseded"), ("in Paris", "active")]
    assert "op=classify" in caplog.text and "Karachi" in caplog.text
    ops = [o for o in mem.store.ops() if o.op_type == "temporal_write"]
    assert [o.failures for o in ops] == [0, 1, 0]


def test_extraction_failure_counted():
    class Boom(Scripted):
        def get_completion(self, prompt, temperature=0.7):
            if prompt.startswith("Extract the atomic facts"):
                raise RuntimeError("down")
            return super().get_completion(prompt, temperature)

    mem = system(Boom())
    note = mem.write("Alice is in Lahore.", "t1")
    assert mem.store.get_note(note) is not None
    assert mem.store.fact_history("s0", "alice", "is") == []
    assert mem.store.ops()[-1].failures == 1


def test_failed_apply_is_rolled_back():
    mem = system(LLM("fake", "fake"))
    mem.write("Alice is in Lahore.", "t1")
    real = mem.consolidator.apply

    def half_done(*args, **kwargs):
        real(*args, **kwargs)  # inserts the new fact and supersedes the old one...
        raise RuntimeError("disk full")  # ...then fails

    mem.consolidator.apply = half_done
    mem.write("Alice is in Karachi.", "t2")
    hist = mem.store.fact_history("s0", "alice", "is")
    assert [(f.value, f.status) for f in hist] == [("in Lahore", "active")]
    assert hist[0].superseded_by is None
    assert mem.store.ops()[-1].failures == 1


def test_note_version_observed_at_when_no_facts():
    mem = system(Scripted(decision="UPDATE_NEIGHBOR"))
    first = mem.write("Alice is happy.", "t1")  # observed_counter -> 1
    second = mem.write("hello there", "t2")     # no facts, rewrites note 1
    (v,) = mem.store.note_versions(first)
    assert v["observed_at"] == 1
    mem.write("Bob is here.", "t3")             # fact observed_at 2, rewrites note 2
    assert [r["observed_at"] for r in mem.store.note_versions(second)] == [2]


def test_fallback_to_add_warns(caplog):
    store = Store()
    with caplog.at_level(logging.WARNING, logger="tsom.consolidator"):
        f = Consolidator(store).apply("s0", "n", {"entity": "E", "attribute": "a", "value": "v"},
                                      "REPLACE", [], 1, None)
    assert f.operation == "ADD" and "falling back to ADD" in caplog.text


def test_cost_table_has_temporal_columns():
    from harness.run_locomo import cost_summary, to_markdown
    ops = [
        {"op_type": "write", "llm_calls": 3, "latency_s": 1.0, "evolved": True},
        {"op_type": "temporal_write", "llm_calls": 2, "latency_s": 0.5, "evolved": None,
         "op_counts": {"ADD": 1}, "failures": 1},
        {"op_type": "write", "llm_calls": 1, "latency_s": 3.0, "evolved": False},
        {"op_type": "temporal_write", "llm_calls": 4, "latency_s": 1.5, "evolved": None,
         "op_counts": {"ADD": 1, "REPLACE": 2}},
        {"op_type": "search", "llm_calls": 0, "latency_s": 9.0, "evolved": None},
    ]
    cost = cost_summary([], ops)
    assert cost["llm_calls_per_write"] == 2 and cost["seconds_per_write"] == 2
    assert cost["temporal_llm_calls_per_write"] == 3 and cost["temporal_seconds_per_write"] == 1
    assert cost["total_llm_calls_per_write"] == 5 and cost["total_seconds_per_write"] == 3
    assert cost["temporal_failures"] == 1 and cost["fact_ops"] == {"ADD": 2, "REPLACE": 2}
    md = to_markdown({"run_name": "r", "system": "tsom", "model": "m", "conversations": 1,
                      "scores": {}, "cost": cost})
    assert "| Per memory write | A-Mem write | Temporal | Total |" in md
    assert "| LLM calls | 2.00 | 3.00 | 5.00 |" in md
    # without temporal ops the cost table is the old one
    plain = cost_summary([], [ops[0], ops[2]])
    assert "total_llm_calls_per_write" not in plain


def test_tsom_run_reports_temporal_cost(tmp_path):
    from harness.run_locomo import run
    config = RunConfig(run_name="tsom_smoke", system="tsom", backend="fake", model="fake",
                       max_samples=1, max_sessions=1, max_questions=3, output_dir=str(tmp_path))
    summary = run(config)
    cost = summary["cost"]
    assert cost["temporal_writes"] == cost["writes"] > 0
    assert cost["total_llm_calls_per_write"] > cost["llm_calls_per_write"]
    assert "Per memory write" in (tmp_path / "tsom_smoke" / "summary.md").read_text()


class Unreadable(Scripted):
    """Scripted extraction; the update classifier's reply cannot be parsed."""

    def get_completion(self, prompt, temperature=0.7):
        if prompt.startswith("A memory already contains this fact:"):
            self.usage.calls += 1
            return "sorry, I am not sure"
        return super().get_completion(prompt, temperature)


def test_unresolved_keeps_the_old_fact_and_is_counted():
    mem = system(Unreadable(facts=[fact_text("Lahore"), fact_text("Karachi")]))
    mem.write("I live in Lahore.", "t1")
    mem.write("I live in Karachi.", "t2")
    assert [(f.value, f.status) for f in mem.store.fact_history("s0", "Alice", "lives_in")] == [("Lahore", "active")]
    ops = [op.model_dump() for op in mem.store.ops()]
    assert ops[-1]["op_counts"] == {"UNRESOLVED": 1} and ops[-1]["failures"] == 0
    assert cost_summary([], ops)["fact_ops"] == {"ADD": 1, "UNRESOLVED": 1}
