"""Validity-annotated retrieval context (Gap 3)."""

import pytest
from pydantic import ValidationError

from agent.assistant import Assistant
from tsom.context import MAX_FACT_LINES, fact_context, retrieve_blocks
from tsom.contracts import Fact, RunConfig
from tsom.llm import LLM
from tsom.memory import MemorySystem
from tsom.rag import RagMemory
from tsom.store import Store
from tsom.tools import MemoryTools

from scripted import ScriptedLLM

S1, S2 = "1:56 pm on 8 May, 2023", "6:30 pm on 14 June, 2023"


def build(*script, use_validity=True, use_classifier=True):
    config = RunConfig(run_name="t", system="tsom", backend="fake", model="fake",
                       use_classifier=use_classifier, use_validity=use_validity)
    llm = ScriptedLLM(script)
    memory = MemorySystem(config, llm)
    return memory, Assistant(MemoryTools(memory), llm)


def karachi_lahore(**flags):
    memory, agent = build(
        "FACT: lives_in | Karachi | additive | none\nFACT: works_at | bank | additive | none",
        "NONE",
        "FACT: lives_in | Lahore | corrective | Karachi",
        **flags,
    )
    n1 = agent.observe("Ali", "I work at a bank in Karachi.", S1)
    n2 = agent.observe("Sara", "Do you enjoy the sea breeze over there?", S1)
    n3 = agent.observe("Ali", "I moved to Lahore last week.", S2)
    return memory, agent, (n1, n2, n3)


def block_of(memory, note_id):
    return dict(retrieve_blocks(memory.amem, "x", 10))[note_id]


# ---- annotated context ----

def test_karachi_to_lahore_context():
    memory, _, (n1, n2, n3) = karachi_lahore()
    result = memory.search("Karachi bank", 10, question="Where does Ali live now?")
    notes = result["notes"]

    assert notes.startswith(
        "Current facts:\n"
        "- Ali lives_in Lahore (since 2023-06-14; previously Karachi until 2023-06-14)\n"
        "- Ali works_at bank (since 2023-05-08)\n\n")
    marker = "[OUTDATED as of 2023-06-14: lives_in Karachi was replaced by Lahore]\n"
    assert block_of(memory, n1) + marker in notes
    assert notes.count("[OUTDATED") == notes.count(block_of(memory, n1))  # only after the session-1 note
    assert block_of(memory, n3) + "[OUTDATED" not in notes

    facts = result["facts"]
    lives = next(g for g in facts["current"] if g["predicate"] == "lives_in")
    assert lives["values"][0]["fact"]["object"] == "Lahore"
    assert [p["object"] for p in lives["values"][0]["previous"]] == ["Karachi"]
    assert {(m["note_id"], m["replaced_by"]) for m in facts["outdated"]} == {(n1, "Lahore")}


def test_note_with_two_facts_one_retired():
    memory, _, (n1, _, _) = karachi_lahore()
    notes = memory.search("Karachi bank", 1, question="Ali")["notes"]
    after = notes.split(block_of(memory, n1), 1)[1]
    assert after.startswith("[OUTDATED as of 2023-06-14: lives_in Karachi was replaced by Lahore]\n")
    assert "works_at" not in after.split("talk start time:")[0]  # the active bank fact gets no marker


def test_question_names_speaker_missing_from_keywords():
    memory, _, (_, n2, _) = karachi_lahore()
    for note in memory.amem.memories.values():
        note.links = []  # fake evolution may link Sara's note to Ali's
    query = "Do you enjoy the sea breeze over there?"  # Sara's note, which has no facts
    plain = memory.amem.find_related_memories_raw(query, k=1)
    assert plain.startswith(block_of(memory, n2))
    assert memory.search(query, 1)["notes"] == plain  # keywords alone name nobody
    with_q = memory.search(query, 1, question="Where does ali live now?")["notes"]
    assert with_q.startswith("Current facts:\n- Ali lives_in Lahore")
    assert with_q.endswith(plain)


def test_assistant_passes_question():
    memory, agent, _ = karachi_lahore()
    seen = {}
    original = memory.search

    def spy(query, k, question=None):
        seen.update(query=query, question=question)
        return original(query, k, question=question)
    memory.search = spy
    out = agent.answer("Where does Ali live now?", 1, "Lahore")
    assert seen["question"] == "Where does Ali live now?"
    assert out["context"].startswith("Current facts:\n")


def test_person_with_no_facts():
    memory, _, _ = karachi_lahore()
    result = memory.search("sea breeze", 10, question="What does Bob like?")
    assert "Bob" not in result["notes"].split("\n\n")[0]
    # With no facts stored at all, the string is the plain A-Mem one.
    empty, agent = build("NONE", "NONE")
    agent.observe("Bob", "Hello there.", S1)
    agent.observe("Sara", "Hi Bob.", S1)
    assert empty.search("hello", 10, question="What does Bob like?")["notes"] == \
        empty.amem.find_related_memories_raw("hello", k=10)
    assert empty.search("hello", 10)["facts"] == {"current": [], "outdated": []}


def test_validity_off_equals_amem_string_but_keeps_facts():
    memory, _, _ = karachi_lahore(use_validity=False)
    for query in ("Karachi", "Lahore move", "sea breeze", "bank job"):
        result = memory.search(query, 2, question="Where does Ali live now?")
        assert result["notes"] == memory.amem.find_related_memories_raw(query, k=2)
        assert result["facts"]["current"] and result["facts"]["outdated"]
    assert [f.object for f in memory.store.active_facts("Ali", "lives_in")] == ["Lahore"]


def test_parse_failure_does_not_break_search():
    memory, agent = build("garbage", "more garbage")
    agent.observe("Ali", "I live in Lahore.", S1)
    result = memory.search("Lahore", 5, question="Where does Ali live?")
    assert result["notes"] == memory.amem.find_related_memories_raw("Lahore", k=5)
    assert memory.store.ops()[0].parse_failures == 1


def test_amem_and_rag_unchanged():
    amem = MemorySystem(RunConfig(run_name="a", backend="fake", model="fake"), ScriptedLLM())
    amem.write("Speaker Alisays : hi", S1, speaker="Ali")
    assert amem.search("hi", 5, question="Ali?") == \
        {"notes": amem.amem.find_related_memories_raw("hi", k=5), "facts": []}
    rag = RagMemory(RunConfig(run_name="r", system="rag", backend="fake", model="fake"), LLM("fake", "f"))
    rag.write("Speaker Alisays : hi", S1, speaker="Ali")
    assert rag.search("hi", 5, question="Ali?")["facts"] == []


# ---- byte identity with A-Mem, real links ----

def test_retrieve_blocks_byte_identical_to_amem():
    memory = MemorySystem(RunConfig(run_name="a", backend="fake", model="fake"), ScriptedLLM())
    topics = ["bank job in Karachi", "biryani for dinner", "my cat Biscuit", "moving to Lahore",
              "startup offer", "nihari breakfast", "sea breeze walks", "cricket on Sunday",
              "reading novels", "learning guitar", "visiting Islamabad", "sister's wedding"]
    for n, topic in enumerate(topics):
        memory.write(f"Speaker Alisays : Let me tell you about {topic}.", S1)
    notes = list(memory.amem.memories.values())
    # Hand-made links: duplicates, self links, long lists (so the break matters),
    # and neighbours that are also top-k hits.
    links = [[1, 2, 3, 4, 5, 6], [0, 0, 2], [2], [], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
             [6, 6, 6, 6], [5], [11, 1], [8, 8], [0], [3, 4], [10, 9, 8]]
    for note, ls in zip(notes, links):
        note.links = list(ls)

    queries = topics[:6] + ["Karachi Lahore", "food", "cat", "music", "family", "Sunday cricket novels"]
    for k in (1, 2, 3, 5, 10):
        for query in queries:
            blocks = retrieve_blocks(memory.amem, query, k)
            assert "".join(t for _, t in blocks) == memory.amem.find_related_memories_raw(query, k=k)
    # A neighbour that is also a hit appears twice, as in A-Mem.
    ids = [nid for nid, _ in retrieve_blocks(memory.amem, "Karachi Lahore", 10)]
    assert len(ids) > len(set(ids))


def test_retrieve_blocks_empty():
    memory = MemorySystem(RunConfig(run_name="a", backend="fake", model="fake"), ScriptedLLM())
    assert retrieve_blocks(memory.amem, "x", 5) == []


# ---- fact block rules (store only) ----

def add(store, subject, predicate, obj, date="2023-01-01", note=None, **kw):
    fact = Fact(subject=subject, predicate=predicate, object=obj, valid_from=date,
                source_note_id=note, **kw)
    store.add_fact(fact)
    return fact


def test_multi_valued_grouped_on_one_line():
    store = Store()
    add(store, "Ali", "likes", "biryani", note="n1")
    add(store, "Ali", "likes", "nihari")
    lines = [g["line"] for g in fact_context(store, ["n1"], "")["current"]]
    assert lines == ["- Ali likes biryani, nihari"]


def test_conflict_marks_both_sides():
    store = Store()
    karachi = add(store, "Ali", "lives_in", "Karachi", "2023-05-08", note="n1")
    add(store, "Ali", "lives_in", "Lahore", "2023-06-14", conflict_with=karachi.id)
    [g] = fact_context(store, ["n1"], "")["current"]
    assert g["line"] == ("- Ali lives_in Karachi (since 2023-05-08; unconfirmed: conflicts with Lahore), "
                         "Lahore (since 2023-06-14; unconfirmed: conflicts with Karachi)")


def test_negated_fact_line_and_marker():
    store = Store()
    tea = add(store, "Ali", "likes", "tea", note="n1")
    add(store, "Ali", "likes", "biryani")
    store.retire(tea.id, "2023-06-14")
    ctx = fact_context(store, ["n1"], "")
    assert [g["line"] for g in ctx["current"]] == [
        "- Ali likes biryani", "- Ali likes tea (no longer true since 2023-06-14)"]
    assert ctx["outdated"]["n1"][0]["line"] == "[OUTDATED as of 2023-06-14: likes tea is no longer true]"


def test_rejected_facts_are_hidden():
    store = Store()
    old = add(store, "Ali", "lives_in", "Karachi", note="n1")
    new = Fact(subject="Ali", predicate="lives_in", object="Lahore", valid_from="2023-06-14", source_note_id="n2")
    store.supersede(old.id, new)
    store.revert_supersession(new.id)
    ctx = fact_context(store, ["n1", "n2"], "")
    assert [g["line"] for g in ctx["current"]] == ["- Ali lives_in Karachi (since 2023-01-01)"]
    assert ctx["outdated"] == {}


def test_cap_keeps_attached_then_newest():
    store = Store()
    predicates = ["lives_in", "works_at", "job_title", "studies_at", "relationship_status",
                  "partner", "age", "allergic_to", "likes", "dislikes"]
    oldest = None
    for speaker in ("Ali", "Sara"):
        for p in predicates:  # 20 groups, inserted oldest first
            f = add(store, speaker, p, f"{speaker}-{p}", note="n1" if (speaker, p) == ("Ali", "age") else None)
            oldest = oldest or f
    gone = add(store, "Ali", "hobby", "chess")
    store.retire(gone.id, "2023-02-01")  # a negated line, which must not exceed the cap
    current = fact_context(store, ["n1"], "Ali and Sara")["current"]
    assert len(current) == MAX_FACT_LINES
    kept = {(g["subject"], g["predicate"]) for g in current}
    assert ("Ali", "age") in kept  # attached to a retrieved note, though old
    assert all(("Sara", p) in kept for p in predicates)  # the 10 newest
    assert ("Ali", "lives_in") not in kept  # the oldest go first
    assert all(g["kind"] == "active" for g in current)


# ---- config validation ----

@pytest.mark.parametrize("system", ["amem", "rag"])
@pytest.mark.parametrize("flag", ["use_classifier", "use_validity"])
def test_baselines_reject_fact_flags(system, flag):
    with pytest.raises(ValidationError, match="baseline"):
        RunConfig(run_name="x", system=system, **{flag: True})
    RunConfig(run_name="x", system="tsom", **{flag: True})
