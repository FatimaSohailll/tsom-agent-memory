import logging
import sqlite3

import pytest

from tsom.contracts import Fact, RunConfig
from tsom.llm import LLM
from tsom.memory import MemorySystem
from tsom.store import Store
from tsom.temporal.classifier import RULE_PAST_BEFORE_EXISTING, UpdateClassifier
from tsom.temporal.consolidator import Consolidator
from tsom.temporal.dates import effective_start, parse_observed_time
from tsom.temporal.extractor import FactExtractor
from tsom.temporal.parsers import parse_facts
from tsom.temporal.prompts import FACT_EXTRACTION_PROMPT, UPDATE_CLASSIFY_PROMPT, build_extraction_prompt


class Canned:
    def __init__(self, response="OPERATION: REPLACE\nREASON: x"):
        self.response, self.prompts = response, []

    def get_completion(self, prompt, temperature=0.7):
        self.prompts.append(prompt)
        return self.response


class NoLLM:
    def get_completion(self, prompt, temperature=0.7):
        raise AssertionError("LLM must not be called")


def microsoft(store, observed_time="2024-03-15", **kw):
    f = Fact(sample_id="s0", note_id="n1", entity="Ali", attribute="works_at", value="Microsoft",
             observed_at=4, observed_time=observed_time, temporal_status="PRESENT", **kw)
    store.add_fact(f)
    return f


def careem(tense="PAST", **kw):
    return {"entity": "Ali", "attribute": "works_at", "value": "Careem", "valid_from": "2018",
            "valid_until": None, "temporal_status": tense, "source_text": "In 2018 Ali worked at Careem.", **kw}


# ---- 1. schema and store ----

def test_fact_default_and_roundtrip():
    f = Fact(sample_id="s", note_id="n", entity="e", attribute="a", value="v", observed_at=1)
    assert f.temporal_status == "UNKNOWN"
    store = Store()
    g = f.model_copy(update={"fact_id": "x", "temporal_status": "PAST"})
    store.add_fact(g)
    assert store.get_fact("x").temporal_status == "PAST"
    with pytest.raises(ValueError):
        Fact(sample_id="s", note_id="n", entity="e", attribute="a", value="v", observed_at=1,
             temporal_status="LATER")


def test_update_fact_status_leaves_tense_alone():
    store = Store()
    f = Fact(sample_id="s", note_id="n", entity="e", attribute="a", value="v", observed_at=1,
             temporal_status="PAST")
    store.add_fact(f)
    store.update_fact_status(f.fact_id, "superseded", valid_until="2020", superseded_by="z")
    assert store.get_fact(f.fact_id).temporal_status == "PAST"


def test_old_database_is_migrated(tmp_path):
    db = tmp_path / "old.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE facts (fact_id TEXT PRIMARY KEY, sample_id TEXT NOT NULL, note_id TEXT NOT NULL,
            entity TEXT NOT NULL, attribute TEXT NOT NULL, value TEXT NOT NULL,
            entity_key TEXT NOT NULL, attr_key TEXT NOT NULL, observed_at INTEGER NOT NULL,
            observed_time TEXT, valid_from TEXT, valid_until TEXT, status TEXT NOT NULL DEFAULT 'active',
            operation TEXT NOT NULL DEFAULT 'ADD', supersedes TEXT NOT NULL DEFAULT '[]',
            superseded_by TEXT, observation_count INTEGER NOT NULL DEFAULT 1);
        INSERT INTO facts (fact_id, sample_id, note_id, entity, attribute, value, entity_key,
            attr_key, observed_at) VALUES ('old', 's', 'n', 'Ali', 'works_at', 'Google', 'ali', 'works_at', 1);
    """)
    conn.commit()
    conn.close()
    store = Store(str(db))
    assert store.get_fact("old").temporal_status == "UNKNOWN"  # existing row kept, defaulted
    store.add_fact(Fact(sample_id="s", note_id="n", entity="Ali", attribute="works_at",
                        value="Microsoft", observed_at=2, temporal_status="PRESENT"))
    Store(str(db))  # opening twice must not fail
    assert [f.temporal_status for f in store.fact_history("s", "ali", "works_at")] == ["UNKNOWN", "PRESENT"]


# ---- 2. extractor ----

def test_prompt_has_tense_rule_and_examples():
    p = FACT_EXTRACTION_PROMPT
    assert "TENSE: ..." in p and "PAST if the wording says it is over or earlier" in p
    assert "never changes VALID_FROM or VALID_UNTIL" in p
    assert "Example text: In 2018 Ali worked at Careem." in p and "TENSE: PAST" in p
    assert "Example text: Ali now works at Microsoft." in p and "TENSE: PRESENT" in p
    assert build_extraction_prompt("hi").endswith("Text:\nhi")


@pytest.mark.parametrize("tense,expected", [("PAST", "PAST"), ("present", "PRESENT"), ("Unknown", "UNKNOWN"),
                                            ("**PAST**", "PAST"), ("sometime", "UNKNOWN"), ("", "UNKNOWN"),
                                            (None, "UNKNOWN")])
def test_parse_tense(tense, expected):
    line = "" if tense is None else f"TENSE: {tense}\n"
    (fact,) = parse_facts(f"FACT 1:\nENTITY: Ali\nATTRIBUTE: works_at\nVALUE: Careem\n"
                          f"VALID_FROM: 2018\nVALID_UNTIL: NONE\n{line}")
    assert fact["temporal_status"] == expected
    assert fact["valid_from"] == "2018" and fact["valid_until"] is None  # tense never touches dates


def test_parse_tense_does_not_split_blocks():
    resp = ("FACT 1:\nENTITY: A\nATTRIBUTE: x\nVALUE: 1\nTENSE: PAST\n"
            "ENTITY: B\nATTRIBUTE: y\nVALUE: 2\nTENSE: PRESENT")  # headers missing
    assert [(f["entity"], f["temporal_status"]) for f in parse_facts(resp)] == [("A", "PAST"), ("B", "PRESENT")]


def test_fake_backend_emits_new_format():
    (fact,) = FactExtractor(LLM("fake", "fake")).extract("The capital of France is Paris.")
    assert fact["temporal_status"] == "UNKNOWN"
    batch = FactExtractor(LLM("fake", "fake")).extract_batch(["Bob is in Rome."])
    assert batch[0]["temporal_status"] == "UNKNOWN" and batch[0]["source"] == 0


# ---- dates ----

def test_observed_time_and_effective_start():
    assert parse_observed_time("2024-03-15") == (2024, 3, 15)
    assert parse_observed_time("1:56 pm on 8 May, 2023") == (2023, 5, 8)
    assert parse_observed_time("June 2023") == (2023, 6)
    assert parse_observed_time("facts 0-19") is None  # no year -> never a date
    assert parse_observed_time("sometime in 2020s soon 99999") is None or True  # must not raise
    assert parse_observed_time(None) is None and parse_observed_time("") is None
    f = Fact(sample_id="s", note_id="n", entity="e", attribute="a", value="v", observed_at=1,
             observed_time="2024-03-15")
    assert effective_start(f) == (2024, 3, 15)
    assert effective_start(f.model_copy(update={"valid_from": "2019"})) == (2019,)  # stated date wins


# ---- 3. deterministic rule ----

def test_a_past_fact_before_existing_is_historical_by_rule():
    store = Store()
    ms = microsoft(store)
    c = UpdateClassifier(NoLLM(), store)
    op, old = c.classify("s0", careem(), 8, "2024-06-01")
    assert op == "ADD_HISTORICAL" and [f.fact_id for f in old] == [ms.fact_id]
    assert c.last_decision["source"] == "past_before_existing"
    assert c.last_decision["reason"] == RULE_PAST_BEFORE_EXISTING

    new = Consolidator(store).apply("s0", "n8", careem(), op, old, 8, "2024-06-01")
    now = store.get_fact(ms.fact_id)
    assert (now.status, now.valid_until, now.superseded_by) == ("active", None, None)  # unchanged
    assert (new.status, new.superseded_by, new.temporal_status) == ("superseded", ms.fact_id, "PAST")
    assert new.valid_until is None and new.valid_from == "2018"  # nothing invented
    assert [f.fact_id for f in store.active_facts("s0", "ali", "works_at")] == [ms.fact_id]


def test_a_end_to_end_through_memory_system():
    class Scripted(LLM):
        def __init__(self):
            super().__init__("fake", "fake")
            self.replies = [
                "FACT 1:\nENTITY: Ali\nATTRIBUTE: works_at\nVALUE: Microsoft\nVALID_FROM: NONE\nVALID_UNTIL: NONE\nTENSE: PRESENT",
                "FACT 1:\nENTITY: Ali\nATTRIBUTE: works_at\nVALUE: Careem\nVALID_FROM: 2018\nVALID_UNTIL: NONE\nTENSE: PAST"]

        def get_completion(self, prompt, temperature=0.7):
            if prompt.startswith("Extract the atomic facts stated in the text."):
                return self.replies.pop(0)
            if prompt.startswith("A memory already contains this fact:"):
                raise AssertionError("the rule should decide this without the LLM")
            return super().get_completion(prompt, temperature)

    mem = MemorySystem(RunConfig(run_name="t", system="tsom", backend="fake", model="fake"),
                       Scripted(), sample_id="s0")
    mem.write("Ali now works at Microsoft.", "2024-03-15")
    mem.write("In 2018 Ali worked at Careem.", "2024-06-01")
    ms, cm = mem.store.fact_history("s0", "ali", "works_at")
    assert (ms.value, ms.status, ms.valid_until) == ("Microsoft", "active", None)
    assert (cm.value, cm.status, cm.superseded_by) == ("Careem", "superseded", ms.fact_id)
    assert dict(mem.classifier.op_counts) == {"ADD": 1, "ADD_HISTORICAL": 1}
    assert "Careem" not in mem.search("Where does Ali work?", 5)["facts"]


@pytest.mark.parametrize("tense", ["UNKNOWN", "PRESENT"])
def test_b_not_past_tense_does_not_trigger_rule(tense):
    store = Store()
    microsoft(store)
    llm = Canned("OPERATION: REPLACE\nREASON: joined later")
    c = UpdateClassifier(llm, store)
    new = careem(tense, source_text="In 2018 Ali joined Careem.")
    op, _ = c.classify("s0", new, 8, "2024-06-01")
    assert op == "REPLACE" and len(llm.prompts) == 1  # went to the LLM path
    assert c.last_decision["source"] == "LLM" and c.last_decision["reason"] == "joined later"
    assert c.last_decision["raw"].startswith("OPERATION: REPLACE")
    assert "tense: " + tense in llm.prompts[0]


def test_b_tense_alone_and_missing_dates_do_not_trigger():
    store = Store()
    microsoft(store)
    llm = Canned()
    c = UpdateClassifier(llm, store)
    no_date = {**careem(), "valid_from": None}
    assert c.classify("s0", no_date, 8)[0] == "REPLACE"            # PAST but no stated date
    later = {**careem(), "valid_from": "2025"}
    assert c.classify("s0", later, 8)[0] == "REPLACE"              # PAST, starts after the existing one
    same_year = {**careem(), "valid_from": "2024"}
    assert c.classify("s0", same_year, 8)[0] == "REPLACE"          # 2024 is not before 2024-03-15
    assert len(llm.prompts) == 3


def test_rule_needs_a_known_start_for_every_old_fact():
    store = Store()
    microsoft(store, observed_time="a while ago")                  # unparseable, no valid_from
    llm = Canned()
    c = UpdateClassifier(llm, store)
    assert c.classify("s0", careem(), 8)[0] == "REPLACE" and len(llm.prompts) == 1


def test_c_serial_facts_never_trigger_the_rule():
    store = Store()
    old = Fact(sample_id="s0", note_id="n", entity="France", attribute="capital_of", value="Lyon",
               observed_at=12, observed_time="facts 0-19")
    store.add_fact(old)
    llm = Canned("OPERATION: REPLACE\nREASON: newer serial")
    c = UpdateClassifier(llm, store)
    new = {"entity": "France", "attribute": "capital_of", "value": "Paris", "valid_from": None,
           "valid_until": None, "temporal_status": "UNKNOWN", "source_text": "40. The capital of France is Paris."}
    op, found = c.classify("s0", new, 40, "facts 40-59")
    assert op == "REPLACE" and found[0].fact_id == old.fact_id and c.last_decision["source"] == "LLM"
    # even mislabelled PAST with a stray date, a non-date label like "facts 0-19" gives no start
    assert c.classify("s0", {**new, "temporal_status": "PAST", "valid_from": "1800"}, 41)[0] == "REPLACE"
    # and the consolidator applies it exactly as before
    Consolidator(store).apply("s0", "n2", new, "REPLACE", found, 40, "facts 40-59")
    assert store.get_fact(old.fact_id).status == "superseded"


def test_existing_rule_with_both_valid_from_still_works():
    store = Store()
    microsoft(store, valid_from="2022")
    c = UpdateClassifier(NoLLM(), store)
    op, _ = c.classify("s0", {**careem("UNKNOWN"), "valid_from": "2019"}, 8)
    assert op == "ADD_HISTORICAL"  # tense is irrelevant here, as before


# ---- 4. classifier prompt ----

def test_classifier_prompt_content():
    p = UPDATE_CLASSIFY_PROMPT
    assert "A past-tense fact about an earlier period does not replace a current fact." in p
    assert "(e.g." not in p and "(multi-valued" not in p  # no parenthetical examples in the options
    assert "Relations such as languages spoken, children, memberships and hobbies can hold several " \
           "values at once." in p
    assert ("If the relation normally has one value at a time, choose REPLACE; "
            "if a person or thing can have several, choose ADD.") in p
    assert "If unsure" not in p
    assert "OPERATION: <REPLACE | CORRECT | ADD>\n" in p  # the format line itself carries no examples
    store = Store()
    microsoft(store, valid_from="2022")
    llm = Canned()
    UpdateClassifier(llm, store).classify("s0", {**careem("PAST"), "valid_from": "2023", "valid_until": "2024"},
                                         8, "2024-06-01")
    prompt = llm.prompts[0]
    assert ("Ali | works_at | Microsoft (learned at step 4, session 2024-03-15, "
            "valid from: 2022, valid until: not stated)") in prompt
    assert ("Ali | works_at | Careem (learned at step 8, session 2024-06-01, "
            "valid from: 2023, valid until: 2024, tense: PAST)") in prompt


def test_classifier_reads_only_the_operation_line():
    store = Store()
    microsoft(store)
    llm = Canned("The old value is wrong, CORRECT it.\nREASON: it should be CORRECT\nOPERATION: ADD\n")
    c = UpdateClassifier(llm, store)
    assert c.classify("s0", careem("PRESENT"), 8)[0] == "ADD"
    llm.response = "REASON: this is a CORRECT case"
    assert c.classify("s0", careem("PRESENT"), 8)[0] == "UNRESOLVED"  # no OPERATION line


# ---- 5. consolidator guard ----

def test_d_guard_turns_a_bad_replace_into_add_historical(caplog):
    store = Store()
    ms = microsoft(store)
    with caplog.at_level(logging.WARNING, logger="tsom.consolidator"):
        c = Consolidator(store)
        new = c.apply("s0", "n8", careem("UNKNOWN"), "REPLACE", [ms], 8, "2024-06-01")
    assert c.last_op == "ADD_HISTORICAL"
    assert new.status == "superseded" and new.superseded_by == ms.fact_id and new.valid_until is None
    assert new.supersedes == []
    now = store.get_fact(ms.fact_id)
    assert (now.status, now.valid_until, now.superseded_by) == ("active", None, None)
    assert ms.fact_id in caplog.text and new.fact_id in caplog.text and "ADD_HISTORICAL" in caplog.text


def test_guard_uses_stated_valid_from_too_and_derives_valid_until():
    store = Store()
    old = Fact(sample_id="s0", note_id="n", entity="Ali", attribute="works_at", value="Microsoft",
               observed_at=1, valid_from="2022")
    store.add_fact(old)
    c = Consolidator(store)
    new = c.apply("s0", "n2", careem(), "REPLACE", [old], 2, None)
    assert c.last_op == "ADD_HISTORICAL" and new.valid_until == "2022"  # old.valid_from is stated text
    assert store.get_fact(old.fact_id).status == "active"


def test_guard_leaves_normal_replace_alone():
    store = Store()
    ms = microsoft(store)
    c = Consolidator(store)
    new = c.apply("s0", "n9", {**careem("PRESENT"), "valid_from": "2025"}, "REPLACE", [ms], 9, None)
    assert c.last_op == "REPLACE" and new.status == "active"
    got = store.get_fact(ms.fact_id)
    assert (got.status, got.valid_until, got.superseded_by) == ("superseded", "2025", new.fact_id)
    plain = c.apply("s0", "n10", {"entity": "Ali", "attribute": "works_at", "value": "Meta"},
                    "REPLACE", [new], 10, None)
    assert c.last_op == "REPLACE" and store.get_fact(new.fact_id).valid_until is None  # undated: stays NULL


def test_op_counts_follow_what_was_applied():
    class Scripted(LLM):
        def __init__(self, replies):
            super().__init__("fake", "fake")
            self.replies = list(replies)

        def get_completion(self, prompt, temperature=0.7):
            if prompt.startswith("Extract the atomic facts stated in the text."):
                return self.replies.pop(0)
            if prompt.startswith("A memory already contains this fact:"):
                return "OPERATION: REPLACE\nREASON: model said so"
            return super().get_completion(prompt, temperature)

    fact = "FACT 1:\nENTITY: Ali\nATTRIBUTE: works_at\nVALUE: {}\nVALID_FROM: {}\nVALID_UNTIL: NONE\nTENSE: {}"
    mem = MemorySystem(RunConfig(run_name="t", system="tsom", backend="fake", model="fake"),
                       Scripted([fact.format("Microsoft", "NONE", "UNKNOWN"),
                                 fact.format("Careem", "2018", "UNKNOWN")]), sample_id="s0")
    mem.write("Ali works at Microsoft.", "2024-03-15")
    mem.write("Ali joined Careem in 2018.", "2024-06-01")  # UNKNOWN tense: LLM says REPLACE, guard intervenes
    assert dict(mem.classifier.op_counts) == {"ADD": 1, "ADD_HISTORICAL": 1}
    assert mem.store.ops()[-1].op_counts == {"ADD_HISTORICAL": 1}
