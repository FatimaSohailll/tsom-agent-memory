import pytest

from tsom.contracts import Fact, RunConfig
from tsom.llm import LLM
from tsom.memory import MemorySystem
from tsom.store import Store
from tsom.temporal.classifier import UpdateClassifier
from tsom.temporal.retriever import TemporalRetriever


class Canned:
    def __init__(self, response="OPERATION: REPLACE\nREASON: x"):
        self.response, self.prompts = response, []

    def get_completion(self, prompt, temperature=0.7):
        self.prompts.append(prompt)
        return self.response


class NoLLM:
    def get_completion(self, prompt, temperature=0.7):
        raise AssertionError("LLM must not be called")


def seed(store, attribute, value, observed_at=1, entity="Ali"):
    f = Fact(sample_id="s0", note_id="n", entity=entity, attribute=attribute, value=value,
             observed_at=observed_at)
    store.add_fact(f)
    return f


def new(attribute, value, sentence="", entity="Ali", **kw):
    return {"entity": entity, "attribute": attribute, "value": value, "valid_from": None,
            "valid_until": None, "temporal_status": "UNKNOWN", "source_text": sentence, **kw}


def system(llm, **flags):
    return MemorySystem(RunConfig(run_name="t", system="tsom", backend="fake", model="fake", **flags),
                        llm, sample_id="s0")


class Scripted(LLM):
    def __init__(self, facts, classify=None):
        super().__init__("fake", "fake")
        self.facts, self.classify = list(facts), classify

    def get_completion(self, prompt, temperature=0.7):
        if prompt.startswith("Extract the atomic facts stated in the text."):
            return self.facts.pop(0)
        if prompt.startswith("A memory already contains this fact:"):
            if self.classify is None:
                raise AssertionError("a rule should have decided this without the LLM")
            return self.classify
        return super().get_completion(prompt, temperature)


def fact(entity, attribute, value, tense="PRESENT"):
    return (f"FACT 1:\nENTITY: {entity}\nATTRIBUTE: {attribute}\nVALUE: {value}\n"
            f"VALID_FROM: NONE\nVALID_UNTIL: NONE\nTENSE: {tense}")


# ---- rules in isolation ----

def test_flags_default_off():
    c = RunConfig(run_name="x")
    assert c.seed_multivalued is False and c.additive_cue is False
    assert c.multivalued_attributes == ["speaks", "languages", "children", "siblings", "hobbies", "member_of"]


def test_learned_multivalued_needs_two_active_facts_and_no_flag():
    store = Store()
    a = seed(store, "speaks", "Urdu")
    llm = Canned()
    c = UpdateClassifier(llm, store)  # both flags off
    assert c.classify("s0", new("speaks", "English"), 2)[0] == "REPLACE"  # one active value: ask the LLM
    assert len(llm.prompts) == 1
    b = seed(store, "speaks", "English", 2)
    c2 = UpdateClassifier(NoLLM(), store)
    op, old = c2.classify("s0", new("speaks", "French"), 3)
    assert op == "ADD" and [f.fact_id for f in old] == [a.fact_id, b.fact_id]  # touched, stay active
    assert c2.last_decision["source"] == "learned_multivalued"
    assert c2.last_decision["reason"] == "rule: key already holds several active values"
    assert c2.last_decision["raw"] is None


def test_seed_rule_is_flagged_and_exact():
    store = Store()
    seed(store, "speaks", "Urdu")
    seed(store, "plays_for", "Lahore FC", entity="Ali")
    assert UpdateClassifier(Canned(), store).classify("s0", new("speaks", "English"), 2)[0] == "REPLACE"  # off
    on = UpdateClassifier(NoLLM(), store, seed_multivalued=True)
    op, old = on.classify("s0", new("Speaks", "English"), 2)  # case-insensitive, like every key
    assert op == "ADD" and len(old) == 1 and on.last_decision["source"] == "seed_multivalued"
    # exact match only: near names and substrings do not count
    llm = Canned()
    near = UpdateClassifier(llm, store, seed_multivalued=True)
    store2 = Store()
    for attr in ("speaks_to", "languages_spoken", "children_of", "member", "plays_for"):
        seed(store2, attr, "old")
        near.store = store2
        assert near.classify("s0", new(attr, "new"), 2)[0] == "REPLACE", attr
    assert len(llm.prompts) == 5
    # the list is configurable
    custom = UpdateClassifier(NoLLM(), store, seed_multivalued=True, multivalued_attributes=["plays_for"])
    assert custom.classify("s0", new("plays_for", "Karachi FC"), 3)[0] == "ADD"


@pytest.mark.parametrize("sentence,fires", [
    ("Ali also speaks English.", True),
    ("Ali speaks English as well.", True),
    ("Ali speaks English too.", True),
    ("In addition, Ali speaks English.", True),
    ("ALI ALSO SPEAKS ENGLISH.", True),
    ("Ali also moved to Lahore.", False),
    ("Ali also switched to English.", False),
    ("Ali also no longer speaks Urdu.", False),
    ("Ali speaks English instead.", False),
    ("Ali also changed his job.", False),
    ("Ali left Google too.", False),
    ("Ali also formerly lived there.", False),
    ("Ali also now lives there.", False),
    ("Ali speaks English.", False),                 # no additive cue
    ("Ali works in a toolshop.", False),           # "too" only inside a word
])
def test_additive_cue(sentence, fires):
    store = Store()
    seed(store, "speaks", "Urdu")
    llm = Canned()
    c = UpdateClassifier(llm, store, additive_cue=True)
    op, old = c.classify("s0", new("speaks", "English", sentence), 2)
    assert (op == "ADD" and c.last_decision["source"] == "additive_cue") is fires
    assert bool(llm.prompts) is not fires  # fired -> no LLM call
    assert [f.value for f in old] == ["Urdu"]


def test_additive_cue_is_flagged():
    store = Store()
    seed(store, "speaks", "Urdu")
    llm = Canned()
    assert UpdateClassifier(llm, store).classify("s0", new("speaks", "English", "Ali also speaks English."), 2)[0] == "REPLACE"
    assert len(llm.prompts) == 1


def test_rules_run_after_the_historical_rules():
    store = Store()
    seed(store, "works_at", "Microsoft")
    store.conn.execute("UPDATE facts SET observed_time = '2024-03-15'")
    c = UpdateClassifier(NoLLM(), store, seed_multivalued=True, additive_cue=True,
                         multivalued_attributes=["works_at"])
    op, _ = c.classify("s0", new("works_at", "Careem", "Ali also worked at Careem.",
                                 valid_from="2018", temporal_status="PAST"), 8, "2024-06-01")
    assert op == "ADD_HISTORICAL" and c.last_decision["source"] == "past_before_existing"


def test_decision_sources_are_rule_names():
    store = Store()
    c = UpdateClassifier(Canned(), store)
    c.classify("s0", new("speaks", "Urdu"), 1)
    assert c.last_decision["source"] == "no_existing_fact"
    seed(store, "speaks", "Urdu")
    c.classify("s0", new("speaks", "urdu."), 2)
    assert c.last_decision["source"] == "same_value"
    c.classify("s0", new("speaks", "English"), 3)
    assert c.last_decision["source"] == "LLM" and c.last_decision["raw"]


# ---- end to end ----

def test_a_additive_cue_urdu_then_english():
    llm = Scripted([fact("Ali", "speaks", "Urdu"), fact("Ali", "speaks", "English")])
    mem = system(llm, additive_cue=True)
    mem.write("Ali speaks Urdu.", "2024-05-01")
    mem.write("Ali also speaks English.", "2024-05-02")
    assert mem.classifier.last_decision["source"] == "additive_cue"
    urdu, english = mem.store.fact_history("s0", "ali", "speaks")
    assert (urdu.status, english.status, english.operation) == ("active", "active", "ADD")
    assert urdu.superseded_by is None and english.supersedes == []
    facts = mem.search("What languages does Ali speak?", 5)["facts"].splitlines()
    assert "[CURRENT] Ali | speaks | Urdu (step 1)" in facts and "[CURRENT] Ali | speaks | English (step 2)" in facts
    assert dict(mem.classifier.op_counts) == {"ADD": 2}


def test_b_also_moved_must_not_fire_additive_rule():
    llm = Scripted([fact("Ali", "located_in", "Karachi"), fact("Ali", "located_in", "Lahore")],
                   classify="OPERATION: REPLACE\nREASON: he relocated")
    mem = system(llm, additive_cue=True)
    mem.write("Ali lives in Karachi.", "t1")
    mem.write("Ali also moved to Lahore.", "t2")
    assert mem.classifier.last_decision["source"] == "LLM"
    karachi, lahore = mem.store.fact_history("s0", "ali", "located_in")
    assert (karachi.status, lahore.status) == ("superseded", "active")


def test_c_flags_off_llm_add_then_third_language_by_learned_rule():
    llm = Scripted([fact("Ali", "speaks", "Urdu"), fact("Ali", "speaks", "English"),
                    fact("Ali", "speaks", "French")], classify="OPERATION: ADD\nREASON: several languages")
    mem = system(llm)  # both flags off
    assert mem.config.seed_multivalued is False and mem.config.additive_cue is False
    mem.write("Ali speaks Urdu.", "t1")
    mem.write("Ali speaks English.", "t2")
    assert mem.classifier.last_decision["source"] == "LLM"  # step 7: the LLM said ADD
    llm.classify = None  # from here a classify call would raise
    mem.write("Ali speaks French.", "t3")
    assert mem.classifier.last_decision["source"] == "learned_multivalued"
    assert {f.value for f in mem.store.active_facts("s0", "ali", "speaks")} == {"Urdu", "English", "French"}
    assert dict(mem.classifier.op_counts) == {"ADD": 3}


def test_d_serial_facts_flags_off_behave_as_before():
    mem = system(LLM("fake", "fake"))
    mem.write("0. Alice is in Lahore.\n3. Alice is in Karachi.\n5. Alice is in Paris.\n6. Bob is in Rome.\n"
              "9. Bob is in Oslo.", "facts 0-9")
    for who, n_values in (("alice", 3), ("bob", 2)):
        hist = mem.store.fact_history("s0", who, "is")
        assert len(hist) == n_values
        assert [f.status for f in hist] == ["superseded"] * (n_values - 1) + ["active"]  # exactly one active
        assert [f.operation for f in hist] == ["ADD"] + ["REPLACE"] * (n_values - 1)
    assert dict(mem.classifier.op_counts) == {"ADD": 2, "REPLACE": 3}
    assert mem.classifier.last_decision["source"] == "LLM"  # conflicts went to the model, not a rule


# ---- retriever: current queries ----

def retrieval(store_facts):
    mem = system(LLM("fake", "fake"))
    for attribute, value, status, observed_at in store_facts:
        f = Fact(sample_id="s0", note_id="n", entity="Ali", attribute=attribute, value=value,
                 observed_at=observed_at, status=status)
        mem.store.add_fact(f)
    return mem.temporal_retriever.search("Ali", 5)["facts"].splitlines()


def test_e_single_active_fact_returns_one():
    lines = retrieval([("works_at", "Google", "superseded", 1), ("works_at", "Microsoft", "active", 4),
                       ("works_at", "Meta", "superseded", 3)])
    assert lines == ["[CURRENT] Ali | works_at | Microsoft (step 4)"]


def test_e_several_active_facts_all_returned():
    lines = retrieval([("speaks", "Urdu", "active", 6), ("speaks", "English", "active", 7),
                       ("speaks", "French", "superseded", 5), ("speaks", "Latin", "retracted", 4)])
    assert lines == ["[CURRENT] Ali | speaks | Urdu (step 6)", "[CURRENT] Ali | speaks | English (step 7)"]


def test_e_historical_queries_unchanged():
    mem = system(LLM("fake", "fake"))
    for value, status, obs in [("Urdu", "active", 6), ("English", "active", 7), ("French", "superseded", 5)]:
        mem.store.add_fact(Fact(sample_id="s0", note_id="n", entity="Ali", attribute="speaks", value=value,
                                observed_at=obs, status=status))
    lines = mem.temporal_retriever.search("Ali speaks previously", 5)["facts"].splitlines()
    assert [l.split("|")[2].split("(")[0].strip() for l in lines] == ["French", "Urdu", "English"]
    assert lines[0].startswith("[PAST]")
