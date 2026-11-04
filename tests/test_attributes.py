import numpy as np
import pytest

from tsom.contracts import Fact, RunConfig
from tsom.llm import LLM
from tsom.memory import MemorySystem
from tsom.store import Store
from tsom.temporal.attributes import AttributeCanonicalizer, canonical_form
from tsom.temporal.extractor import FactExtractor
from tsom.temporal.prompts import build_batch_extraction_prompt, build_extraction_prompt


def add(store, entity, attribute, value="v", n=1, sample="s0"):
    for i in range(n):
        store.add_fact(Fact(sample_id=sample, note_id="n", entity=entity, attribute=attribute,
                            value=f"{value}{i}", observed_at=i))


# ---- canonical form ----

@pytest.mark.parametrize("a,b", [("speaks", "speaks_language"), ("speaks", "Speaks"),
                                 ("capital_of", "capital"), ("head_of_state", "head_of_state_of"),
                                 ("spouse", "spouse_of"), ("works_at", "works_at"), ("lives_in", "lives-in")])
def test_same_canonical_form(a, b):
    assert canonical_form(a) == canonical_form(b)


@pytest.mark.parametrize("a,b", [("speaks", "speaks_to"), ("plays_position", "plays_for"),
                                 ("born_in", "died_in"), ("head_of_state", "head_of_government"),
                                 ("works_at", "works_for"), ("works_at", "located_in"),
                                 ("member_of", "founder_of"), ("plays_position", "position")])
def test_sibling_relations_stay_apart(a, b):
    assert canonical_form(a) != canonical_form(b)


# ---- snapping ----

def test_canonical_form_snap_is_entity_scoped_and_keeps_original():
    store = Store()
    add(store, "Ali", "speaks")
    add(store, "Sara", "speaks_language")
    c = AttributeCanonicalizer(store)
    assert c.canonicalize("s0", "Ali", "speaks_language") == ("speaks", "same canonical form")
    assert c.canonicalize("s0", "ali", "Speaks")[1] is None           # same key already
    assert c.canonicalize("s0", "Ali", "works_at") == ("works_at", None)
    assert c.canonicalize("s0", "Nobody", "speaks_language") == ("speaks_language", None)  # no history
    assert c.canonicalize("other", "Ali", "speaks_language")[1] is None                   # other sample
    assert c.snaps["canonical"] == 1


def test_prefers_the_most_used_existing_spelling():
    store = Store()
    add(store, "Ali", "speaks", n=1)
    add(store, "Ali", "speaks_language", n=3)  # same canonical form, used more often
    c = AttributeCanonicalizer(store)
    assert c.canonicalize("s0", "Ali", "languages_speak") == ("speaks_language", "same canonical form")


class StubEncoder:
    """Vectors chosen by hand: cosine between 'x' and 'y' is 0.9, 'z' is 0.5."""
    V = {"x": [1, 0], "y": [0.9, np.sqrt(1 - 0.81)], "z": [0.5, np.sqrt(1 - 0.25)]}

    def __call__(self, texts):
        return np.array([self.V[t] for t in texts])


def test_embedding_snap_respects_threshold():
    store = Store()
    add(store, "Ali", "x")
    assert AttributeCanonicalizer(store, StubEncoder(), threshold=0.88).canonicalize("s0", "Ali", "y") == (
        "x", "embedding similarity 0.90")
    assert AttributeCanonicalizer(store, StubEncoder(), threshold=0.95).canonicalize("s0", "Ali", "y") == ("y", None)
    assert AttributeCanonicalizer(store, StubEncoder(), threshold=0.88).canonicalize("s0", "Ali", "z") == ("z", None)
    assert AttributeCanonicalizer(store, None).canonicalize("s0", "Ali", "y") == ("y", None)  # stage off


def test_real_embedder_never_merges_sibling_relations():
    mem = MemorySystem(RunConfig(run_name="t", system="tsom", backend="fake", model="fake"),
                       LLM("fake", "fake"), sample_id="s0")
    c = mem.attributes
    for known, new in [("born_in", "died_in"), ("head_of_state", "head_of_government"),
                       ("speaks", "speaks_to"), ("died_in", "buried_in"), ("plays_position", "plays_for"),
                       ("works_at", "studied_at"), ("author_of", "director_of")]:
        add(mem.store, f"E_{known}", known)
        assert c.canonicalize("s0", f"E_{known}", new) == (new, None), (known, new)


# ---- the hint ----

def test_known_attributes_entity_first_then_frequent():
    store = Store()
    add(store, "Ali", "speaks")
    add(store, "Ali", "works_at", n=2)
    add(store, "France", "capital_of", n=5)
    add(store, "Rome", "located_in", n=3)
    c = AttributeCanonicalizer(store)
    assert c.known_attributes("s0", "Ali lives somewhere") == ["works_at", "speaks", "capital_of", "located_in"]
    assert c.known_attributes("s0", "unrelated text") == ["capital_of", "located_in", "works_at", "speaks"]
    assert AttributeCanonicalizer(store, hint_limit=2).known_attributes("s0", "Ali") == ["works_at", "speaks"]
    assert AttributeCanonicalizer(Store()).known_attributes("s0", "Ali") == []
    assert c.known_attributes("other", "Ali") == []
    add(store, "Al", "short")  # "al" is in "Ali" but not as a whole word
    assert c.known_attributes("s0", "Ali")[0] == "works_at"


def test_prompt_hint():
    p = build_extraction_prompt("Ali speaks Urdu.", ["speaks", "works_at"])
    assert "Attribute names already in use (reuse one of them if it means the same relation" in p
    assert "speaks, works_at\n\nText:\nAli speaks Urdu." in p and p.endswith("Ali speaks Urdu.")
    assert "Attribute names already in use" not in build_extraction_prompt("hi")
    assert build_extraction_prompt("hi", []).endswith("Text:\nhi")
    b = build_batch_extraction_prompt("1. hi", ["speaks"])
    assert "speaks\n\nStatements:\n1. hi" in b and "Attribute names already in use" not in \
        build_batch_extraction_prompt("1. hi")


def test_extractor_passes_hint_through():
    class Canned:
        def __init__(self):
            self.prompts = []

        def get_completion(self, prompt, temperature=0.7):
            self.prompts.append(prompt)
            return "NO_FACTS"

    llm = Canned()
    ex = FactExtractor(llm)
    ex.extract("Ali speaks Urdu.", ["speaks"])
    ex.extract_batch(["Ali speaks Urdu."], ["speaks"])
    ex.extract("Ali speaks Urdu.")
    assert "speaks\n\nText:" in llm.prompts[0] and "speaks\n\nStatements:" in llm.prompts[1]
    assert "Attribute names already in use" not in llm.prompts[2]


# ---- end to end ----

class Scripted(LLM):
    def __init__(self, replies):
        super().__init__("fake", "fake")
        self.replies, self.extraction_prompts = list(replies), []

    def get_completion(self, prompt, temperature=0.7):
        if prompt.startswith("Extract the atomic facts stated in"):
            self.extraction_prompts.append(prompt)
            return self.replies.pop(0)
        return super().get_completion(prompt, temperature)


def fact(entity, attribute, value):
    return (f"FACT 1:\nENTITY: {entity}\nATTRIBUTE: {attribute}\nVALUE: {value}\n"
            "VALID_FROM: NONE\nVALID_UNTIL: NONE\nTENSE: PRESENT")


def test_drift_is_fixed_end_to_end():
    llm = Scripted([fact("Ali", "speaks", "Urdu"), fact("Ali", "speaks_language", "English")])
    mem = MemorySystem(RunConfig(run_name="t", system="tsom", backend="fake", model="fake"), llm, sample_id="s0")
    mem.write("Ali speaks Urdu.", "2024-05-01")
    mem.write("Ali also speaks English.", "2024-05-02")
    assert "Attribute names already in use" not in llm.extraction_prompts[0]   # nothing known yet
    assert "already in use (reuse one of them if it means the same relation, otherwise use a new short " \
           "snake_case name): speaks\n\nText:" in llm.extraction_prompts[1]
    urdu, english = mem.store.fact_history("s0", "ali", "speaks")                # one key now
    assert (urdu.attribute, english.attribute) == ("speaks", "speaks")
    assert english.raw_attribute == "speaks_language" and urdu.raw_attribute is None
    assert (urdu.status, english.status) == ("active", "active")                # fake classifier: "also" -> ADD
    assert mem.store.fact_history("s0", "ali", "speaks_language") == []
    assert {f.value for f in mem.store.active_facts("s0", "ali", "speaks")} == {"Urdu", "English"}
    assert mem.attributes.snaps["canonical"] == 1
    assert "[CURRENT] Ali | speaks | English" in mem.search("What languages does Ali speak?", 5)["facts"]


def test_drifted_replace_now_supersedes():
    llm = Scripted([fact("Ali", "works_at", "Google"), fact("Ali", "work_at", "Microsoft")])
    mem = MemorySystem(RunConfig(run_name="t", system="tsom", backend="fake", model="fake"), llm, sample_id="s0")
    mem.write("Ali works at Google.", "t1")
    mem.write("Ali now works at Microsoft.", "t2")
    google, microsoft = mem.store.fact_history("s0", "ali", "works_at")
    assert (google.status, microsoft.status, microsoft.operation) == ("superseded", "active", "REPLACE")


def test_failure_in_canonicalizer_does_not_lose_the_fact():
    llm = Scripted([fact("Ali", "speaks", "Urdu"), fact("Ali", "speaks_language", "English")])
    mem = MemorySystem(RunConfig(run_name="t", system="tsom", backend="fake", model="fake"), llm, sample_id="s0")
    mem.write("Ali speaks Urdu.", "t1")
    mem.attributes.canonicalize = lambda *a: (_ for _ in ()).throw(RuntimeError("boom"))
    mem.write("Ali also speaks English.", "t2")
    assert {f.value for f in mem.store.active_facts("s0", "ali", "speaks_language")} == {"English"}


def test_raw_attribute_column_migrates(tmp_path):
    import sqlite3
    db = tmp_path / "old.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE facts (fact_id TEXT PRIMARY KEY, sample_id TEXT NOT NULL, note_id TEXT NOT NULL,
            entity TEXT NOT NULL, attribute TEXT NOT NULL, value TEXT NOT NULL,
            entity_key TEXT NOT NULL, attr_key TEXT NOT NULL, observed_at INTEGER NOT NULL,
            observed_time TEXT, valid_from TEXT, valid_until TEXT, status TEXT NOT NULL DEFAULT 'active',
            operation TEXT NOT NULL DEFAULT 'ADD', supersedes TEXT NOT NULL DEFAULT '[]',
            superseded_by TEXT, observation_count INTEGER NOT NULL DEFAULT 1,
            temporal_status TEXT NOT NULL DEFAULT 'UNKNOWN');
        INSERT INTO facts (fact_id, sample_id, note_id, entity, attribute, value, entity_key, attr_key,
            observed_at) VALUES ('old', 's', 'n', 'Ali', 'speaks', 'Urdu', 'ali', 'speaks', 1);
    """)
    conn.commit()
    conn.close()
    store = Store(str(db))
    assert store.get_fact("old").raw_attribute is None
    assert store.attribute_counts("s", "ali") == [("speaks", 1)]
