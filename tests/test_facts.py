"""Fact layer (Gap 1) with a scripted LLM that returns canned extraction lines."""

import pytest

from agent.assistant import Assistant
from tsom.contracts import Fact, RunConfig
from tsom.facts import FactLayer, parse_output
from tsom.gated_amem import GatedAgenticMemory
from tsom.llm import LLM
from tsom.memory import MemorySystem
from tsom.rag import RagMemory
from tsom.store import Store
from tsom.tools import MemoryTools

from scripted import ScriptedLLM


def layer(*script, use_classifier=True, store=None):
    store = store or Store()
    return FactLayer(store, ScriptedLLM(script), use_classifier), store


def add(store, predicate, value, subject="Ali"):
    fact = Fact(subject=subject, predicate=predicate, object=value, valid_from="d0")
    store.add_fact(fact)
    return fact


def active(store, predicate=None):
    return sorted(f.object for f in store.active_facts("Ali", predicate))


# ---- label paths (classifier mode) ----

def test_additive():
    fl, store = layer("FACT: lives_in | Karachi | additive | none")
    ops = fl.process_turn("Ali", "I live in Karachi.", "d1")
    assert [op.label for op in ops] == ["additive"]
    fact = store.get_fact(ops[0].fact_id)
    assert (fact.subject, fact.predicate, fact.object, fact.valid_from) == ("Ali", "lives_in", "Karachi", "d1")
    assert fl.llm.temperatures == [0.0]


def test_redundant_adds_nothing():
    fl, store = layer("FACT: likes | Biryani | redundant | none")
    add(store, "likes", "biryani")
    assert [op.label for op in fl.process_turn("Ali", "Biryani is the best.", "d1")] == ["redundant"]
    assert len(store.fact_history("Ali")) == 1


def test_model_redundant_on_new_value_adds_nothing():
    fl, store = layer("FACT: likes | nihari | redundant | none")
    add(store, "likes", "biryani")
    assert [op.label for op in fl.process_turn("Ali", "...", "d1")] == ["redundant"]
    assert active(store) == ["biryani"]


def test_corrective_single_supersedes():
    fl, store = layer("FACT: lives_in | Lahore | corrective | Karachi")
    old = add(store, "lives_in", "Karachi")
    [op] = fl.process_turn("Ali", "I moved to Lahore.", "d2")
    assert op.label == "corrective" and op.retired_id == old.id
    assert active(store) == ["Lahore"]
    assert store.get_fact(old.id).valid_to == "d2"
    assert store.get_fact(op.fact_id).supersedes == old.id


def test_corrective_single_without_active_is_additive():
    fl, store = layer("FACT: lives_in | Lahore | corrective | Karachi")
    [op] = fl.process_turn("Ali", "I moved to Lahore.", "d2")
    assert op.label == "additive" and op.raw_label == "corrective"
    assert active(store) == ["Lahore"]


def test_contradictory_keeps_both():
    fl, store = layer("FACT: lives_in | Lahore | contradictory | none")
    old = add(store, "lives_in", "Karachi")
    [op] = fl.process_turn("Ali", "Lahore traffic is awful today.", "d2")
    assert op.label == "contradictory" and op.retired_id is None
    assert active(store) == ["Karachi", "Lahore"]
    assert store.get_fact(op.fact_id).conflict_with == old.id


def test_corrective_multi_retires_only_named_value():
    fl, store = layer("FACT: likes | green tea | corrective | coffee")
    add(store, "likes", "coffee")
    add(store, "likes", "biryani")
    [op] = fl.process_turn("Ali", "I switched from coffee to green tea.", "d2")
    assert op.label == "corrective"
    assert active(store) == ["biryani", "green tea"]


def test_corrective_multi_without_match_is_additive():
    fl, store = layer("FACT: likes | green tea | corrective | juice")
    add(store, "likes", "coffee")
    [op] = fl.process_turn("Ali", "...", "d2")
    assert op.label == "additive" and op.retired_id is None
    assert active(store) == ["coffee", "green tea"]


def test_negation_on_multi_valued_retires_only_that_value():
    fl, store = layer("FACT: likes | none | corrective | tea")
    add(store, "likes", "biryani")
    tea = add(store, "likes", "Tea")
    add(store, "has_pet", "cat")
    [op] = fl.process_turn("Ali", "I don't like tea anymore.", "d2")
    assert op.label == "corrective" and op.retired_id == tea.id and op.fact_id is None
    assert active(store) == ["biryani", "cat"]
    assert store.get_fact(tea.id).status == "retired"


def test_negation_without_match_changes_nothing():
    fl, store = layer("FACT: likes | none | corrective | coffee")
    add(store, "likes", "tea")
    assert fl.process_turn("Ali", "I don't like coffee.", "d2") == []
    assert active(store) == ["tea"]


# ---- several active single values ----

def test_corrective_supersedes_matching_old_among_several():
    fl, store = layer("FACT: lives_in | Islamabad | corrective | karachi")
    karachi = add(store, "lives_in", "Karachi")
    lahore = add(store, "lives_in", "Lahore")
    [op] = fl.process_turn("Ali", "I moved from Karachi to Islamabad.", "d3")
    assert op.retired_id == karachi.id
    assert store.get_fact(lahore.id).status == "active"
    assert active(store) == ["Islamabad", "Lahore"]


def test_corrective_supersedes_newest_inserted_when_old_does_not_match():
    fl, store = layer("FACT: lives_in | Islamabad | corrective | Quetta")
    # Inserted Karachi then Lahore; "9 May" > "10 June" as text, so date order would differ.
    karachi = Fact(subject="Ali", predicate="lives_in", object="Karachi", valid_from="10 June")
    lahore = Fact(subject="Ali", predicate="lives_in", object="Lahore", valid_from="9 May")
    store.add_fact(karachi)
    store.add_fact(lahore)
    [op] = fl.process_turn("Ali", "I moved to Islamabad.", "d3")
    assert op.retired_id == lahore.id
    assert active(store) == ["Islamabad", "Karachi"]


# ---- label corrections in classifier mode ----

def test_additive_on_differing_single_becomes_contradictory():
    fl, store = layer("FACT: works_at | startup | additive | none")
    bank = add(store, "works_at", "bank")
    [op] = fl.process_turn("Ali", "...", "d2")
    assert (op.raw_label, op.label, op.conflict_with) == ("additive", "contradictory", bank.id)
    assert active(store) == ["bank", "startup"]


def test_contradictory_on_multi_becomes_additive():
    fl, store = layer("FACT: likes | nihari | contradictory | none")
    add(store, "likes", "biryani")
    [op] = fl.process_turn("Ali", "...", "d2")
    assert op.label == "additive" and store.get_fact(op.fact_id).conflict_with is None


def test_contradictory_without_active_becomes_additive():
    fl, store = layer("FACT: lives_in | Lahore | contradictory | none")
    [op] = fl.process_turn("Ali", "...", "d2")
    assert op.label == "additive" and op.conflict_with is None


# ---- safety net ----

@pytest.mark.parametrize("label", ["additive", "corrective", "contradictory"])
def test_safety_net_same_value_is_redundant(label):
    fl, store = layer(f"FACT: lives_in |  karachi  | {label} | Karachi")
    add(store, "lives_in", "Karachi")
    [op] = fl.process_turn("Ali", "Still in Karachi.", "d2")
    assert op.label == "redundant" and op.raw_label == label
    assert len(store.fact_history("Ali")) == 1 and active(store) == ["Karachi"]


def test_subject_case_is_ignored():
    fl, store = layer("FACT: likes | biryani | additive | none")
    add(store, "likes", "Biryani", subject="ali")
    [op] = fl.process_turn("Ali", "I love biryani.", "d1")
    assert op.label == "redundant"
    assert "likes: Biryani" in fl.llm.prompts[0]  # active facts reach the prompt


# ---- parsing ----

def test_parse_is_lenient():
    text = """Here you go:
  - fact :  Lives In | Lahore. | Corrective. | karachi
1. FACT: likes | "nihari" | ADDITIVE
FACT: favourite_colour | blue | additive | none
FACT: likes | mango | surprising | none
FACT: only | two"""
    facts, said_none = parse_output(text)
    assert not said_none
    assert [(f.predicate, f.value, f.label, f.old) for f in facts] == [
        ("lives_in", "Lahore", "corrective", "karachi"),
        ("likes", "nihari", "additive", None),
    ]


def test_none_is_not_a_failure():
    fl, store = layer("None.")
    assert fl.process_turn("Ali", "Bye!", "d1") == []
    assert fl.parse_failures == 0 and len(fl.llm.prompts) == 1


def test_parse_failure_retries_once_then_counts():
    fl, store = layer("Sure! Ali lives in Lahore.", "lives_in = Lahore")
    assert fl.process_turn("Ali", "I live in Lahore.", "d1") == []
    assert fl.parse_failures == 1 and len(fl.llm.prompts) == 2
    assert fl.llm.usage.calls == 2 and store.fact_history("Ali") == []


def test_retry_can_recover():
    fl, store = layer("garbage", "FACT: lives_in | Lahore | additive | none")
    assert [op.label for op in fl.process_turn("Ali", "I live in Lahore.", "d1")] == ["additive"]
    assert fl.parse_failures == 0 and fl.last_raw == ["garbage", "FACT: lives_in | Lahore | additive | none"]


def test_failed_call_does_not_crash():
    class Broken(ScriptedLLM):
        def _call(self, prompt, temperature):
            raise RuntimeError("down")
    broken = Broken()
    broken.retries = 0
    fl = FactLayer(Store(), broken, use_classifier=True)
    assert fl.process_turn("Ali", "I live in Lahore.", "d1") == []
    assert fl.parse_failures == 1


# ---- newest value wins (use_classifier=False) ----

def test_newest_wins_retires_on_plain_single_change():
    fl, store = layer("FACT: lives_in | Lahore | contradictory | none", use_classifier=False)
    old = add(store, "lives_in", "Karachi")
    [op] = fl.process_turn("Ali", "Lahore traffic is awful today.", "d2")
    assert (op.raw_label, op.label, op.retired_id) == ("contradictory", "corrective", old.id)
    assert active(store) == ["Lahore"]


def test_newest_wins_multi_is_additive_and_never_retires():
    fl, store = layer("FACT: likes | green tea | corrective | coffee",
                      "FACT: likes | none | corrective | biryani", use_classifier=False)
    add(store, "likes", "coffee")
    add(store, "likes", "biryani")
    [op] = fl.process_turn("Ali", "I switched to green tea.", "d2")
    assert op.label == "additive"
    assert fl.process_turn("Ali", "No more biryani.", "d3") == []
    assert active(store) == ["biryani", "coffee", "green tea"]


def test_both_modes_send_the_same_prompt():
    a, store_a = layer("NONE", use_classifier=True)
    b, store_b = layer("NONE", use_classifier=False)
    add(store_a, "lives_in", "Karachi")
    add(store_b, "lives_in", "Karachi")
    a.process_turn("Ali", "I moved.", "d1")
    b.process_turn("Ali", "I moved.", "d1")
    assert a.llm.prompts == b.llm.prompts


# ---- MemorySystem integration ----

def tsom(*script, **flags):
    config = RunConfig(run_name="t", system="tsom", backend="fake", model="fake", **flags)
    llm = ScriptedLLM(script)
    memory = MemorySystem(config, llm)
    return memory, Assistant(MemoryTools(memory), llm)


def test_end_to_end_karachi_to_lahore():
    memory, agent = tsom(
        "FACT: lives_in | Karachi | additive | none\nFACT: works_at | bank | additive | none",
        "FACT: likes | biryani | additive | none\nFACT: has_pet | cat | additive | none",
        "FACT: lives_in | Lahore | corrective | Karachi\nFACT: works_at | startup | corrective | bank",
        "FACT: likes | nihari | additive | none",
        use_classifier=True,
    )
    n1 = agent.observe("Ali", "I work at a bank in Karachi.", "8 May, 2023")
    agent.observe("Ali", "I love biryani, and my cat agrees.", "8 May, 2023")
    n3 = agent.observe("Ali", "I moved to Lahore and switched to a startup.", "10 June, 2023")
    agent.observe("Ali", "I also like nihari now.", "10 June, 2023")

    store = memory.store
    assert active(store) == ["Lahore", "biryani", "cat", "nihari", "startup"]
    history = store.fact_history("ali", "lives_in")
    assert [(f.object, f.status, f.valid_from, f.valid_to) for f in history] == [
        ("Karachi", "retired", "2023-05-08", "2023-06-10"),
        ("Lahore", "active", "2023-06-10", None),
    ]
    assert history[0].source_note_id == n1 and history[1].source_note_id == n3
    assert history[1].supersedes == history[0].id

    # The "Speaker Alisays : " prefix is stripped before extraction; A-Mem gets it unchanged.
    assert "Turn: I work at a bank in Karachi.\n" in memory.llm.prompts[0]
    assert store.get_note(n1).content == "Speaker Alisays : I work at a bank in Karachi."

    writes = [op for op in store.ops() if op.op_type == "write"]
    assert writes[2].labels == ["corrective", "corrective"]
    assert writes[2].raw_labels == ["corrective", "corrective"]
    assert writes[2].ids_touched[0] == n3 and len(writes[2].ids_touched) == 5
    assert all(op.parse_failures == 0 for op in writes)

    # The extraction call counts in the write: one more call than A-Mem alone.
    amem = MemorySystem(RunConfig(run_name="a", backend="fake", model="fake"), ScriptedLLM())
    amem.write("Speaker Alisays : I work at a bank in Karachi.", "8 May, 2023", speaker="Ali")
    assert writes[0].llm_calls == amem.store.ops()[0].llm_calls + 1


def test_tsom_logs_parse_failures():
    memory, agent = tsom("not a fact line", "still not")
    agent.observe("Ali", "I live in Lahore.", "d1")
    [write] = [op for op in memory.store.ops() if op.op_type == "write"]
    assert write.parse_failures == 1 and write.labels == []


def test_amem_unchanged():
    llm = ScriptedLLM()
    memory = MemorySystem(RunConfig(run_name="a", backend="fake", model="fake"), llm)
    assert memory.facts is None
    note_id = Assistant(MemoryTools(memory), llm).observe("Ali", "I moved to Lahore.", "d1")
    assert llm.prompts == []  # no extraction call
    assert memory.store.fact_history("Ali") == []
    assert memory.store.get_note(note_id).content == "Speaker Alisays : I moved to Lahore."
    [write] = memory.store.ops()
    assert write.labels == [] and write.raw_labels == [] and write.parse_failures == 0


def test_rag_accepts_speaker():
    memory = RagMemory(RunConfig(run_name="r", system="rag", backend="fake", model="fake"),
                       LLM("fake", "fake"))
    note_id = memory.write("Speaker Alisays : hi", "d1", speaker="Ali")
    assert memory.store.get_note(note_id).content == "Speaker Alisays : hi"
    assert memory.store.ops()[0].llm_calls == 0


def test_flags():
    memory, _ = tsom(use_gate=True)
    assert isinstance(memory.amem, GatedAgenticMemory)
    memory, _ = tsom(use_classifier=True, use_validity=True)
    assert memory.facts.use_classifier
    memory, _ = tsom()
    assert memory.facts is not None and not memory.facts.use_classifier
