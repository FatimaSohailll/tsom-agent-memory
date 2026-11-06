import pytest

from tsom.contracts import Fact
from tsom.llm import LLM
from tsom.store import Store
from tsom.temporal.classifier import UpdateClassifier
from tsom.temporal.dates import date_key, is_before
from tsom.temporal.parsers import parse_update_classification


class Canned:
    def __init__(self, response="OPERATION: REPLACE\nREASON: x"):
        self.response, self.prompts = response, []

    def get_completion(self, prompt, temperature=0.7):
        self.prompts.append(prompt)
        return self.response


class NoLLM:
    def get_completion(self, prompt, temperature=0.7):
        raise AssertionError("LLM must not be called")


def seed(store, value="Lahore", observed_at=1, **kw):
    f = Fact(sample_id="s0", note_id="n1", entity="Alice", attribute="lives_in",
             value=value, observed_at=observed_at, **kw)
    store.add_fact(f)
    return f


def new(value="Karachi", **kw):
    return {"entity": "alice", "attribute": "Lives_In", "value": value,
            "valid_from": None, "valid_until": None, **kw}


def test_add_when_no_old():
    c = UpdateClassifier(NoLLM(), Store())
    assert c.classify("s0", new(), 5) == ("ADD", [])


def test_reaffirm_without_llm():
    store = Store()
    a, b = seed(store, "Paris", 1), seed(store, "Lahore", 2)
    c = UpdateClassifier(NoLLM(), store)
    op, old = c.classify("s0", new("  lahore. "), 3)
    assert op == "REAFFIRM" and [f.fact_id for f in old] == [b.fact_id]


@pytest.mark.parametrize("response,expected", [
    ("OPERATION: REPLACE\nREASON: moved", "REPLACE"),
    ("OPERATION: CORRECT\nREASON: wrong", "CORRECT"),
    ("**OPERATION**: add\nREASON: multi", "ADD"),
])
def test_llm_outcomes(response, expected):
    store = Store()
    a = seed(store)
    llm = Canned(response)
    c = UpdateClassifier(llm, store)
    op, old = c.classify("s0", new(source_text="She moved to Karachi."), 2)
    assert op == expected
    assert [f.fact_id for f in old] == [a.fact_id]  # ADD touches the old fact but leaves it active
    p = llm.prompts[0]
    assert "Alice | lives_in | Lahore (learned at step 1, session unknown, " \
           "valid from: not stated, valid until: not stated)" in p
    assert "alice | Lives_In | Karachi (learned at step 2, session unknown, " \
           "valid from: not stated, valid until: not stated, tense: UNKNOWN)" in p
    assert "She moved to Karachi." in p


@pytest.mark.parametrize("response", ["", "garbage", "OPERATION: maybe\nREASON: unsure",
                                      "REASON: it should CORRECT this"])
def test_unparseable_is_unresolved(response):
    store = Store()
    seed(store)
    c = UpdateClassifier(Canned(response), store)
    op, old = c.classify("s0", new(), 2)
    assert op == "UNRESOLVED" and len(old) == 1
    assert c.op_counts["UNRESOLVED"] == 1 and c.last_decision["raw"] == response


def test_fake_backend():
    store = Store()
    seed(store)
    c = UpdateClassifier(LLM("fake", "fake"), store)
    assert c.classify("s0", new(source_text="Actually, she lives in Karachi."), 2)[0] == "CORRECT"
    assert c.classify("s0", new(source_text="She also lives in Karachi."), 2)[0] == "ADD"
    assert c.classify("s0", new(source_text="She lives in Karachi."), 2)[0] == "REPLACE"


def test_out_of_order_historical_without_llm():
    store = Store()
    a = seed(store, "Lahore", 5, valid_from="2022-03")
    c = UpdateClassifier(NoLLM(), store)
    op, old = c.classify("s0", new("Paris", valid_from="March 2019"), 6)
    assert op == "ADD_HISTORICAL" and [f.fact_id for f in old] == [a.fact_id]


def test_not_historical_cases_use_llm():
    store = Store()
    seed(store, "Lahore", 5, valid_from="2022")
    llm = Canned()
    c = UpdateClassifier(llm, store)
    assert c.classify("s0", new(valid_from="2023"), 6)[0] == "REPLACE"      # later
    assert c.classify("s0", new(valid_from="soon"), 6)[0] == "REPLACE"      # unparseable
    assert c.classify("s0", new(valid_from=None), 6)[0] == "REPLACE"        # unknown
    assert c.classify("s0", new(valid_from="2022-05"), 6)[0] == "REPLACE"   # not provably earlier
    assert len(llm.prompts) == 4
    store2 = Store()
    seed(store2, "Lahore", 5)  # old valid_from unknown
    llm2 = Canned()
    assert UpdateClassifier(llm2, store2).classify("s0", new(valid_from="2001"), 6)[0] == "REPLACE"
    assert len(llm2.prompts) == 1


def test_historical_needs_every_old_after():
    store = Store()
    seed(store, "Lahore", 5, valid_from="2022")
    seed(store, "Karachi", 6, valid_from="2018")
    llm = Canned()
    c = UpdateClassifier(llm, store)
    op, _ = c.classify("s0", new("Paris", valid_from="2020"), 7)
    # not before every old fact, so not ADD_HISTORICAL; two active values -> learned multi-valued
    assert op == "ADD" and c.last_decision["source"] == "learned_multivalued" and not llm.prompts


def test_op_counts():
    store = Store()
    c = UpdateClassifier(Canned(), store)
    c.classify("s0", new(), 1)
    seed(store)
    c.classify("s0", new("Lahore"), 2)
    c.classify("s0", new("Paris"), 3)
    assert dict(c.op_counts) == {"ADD": 1, "REAFFIRM": 1, "REPLACE": 1}


def test_other_sample_ignored():
    store = Store()
    seed(store)
    assert UpdateClassifier(NoLLM(), store).classify("other", new(), 1) == ("ADD", [])


def test_dates():
    assert date_key("2020-05-17") == (2020, 5, 17)
    assert date_key("17 May 2020") == date_key("May 17, 2020") == (2020, 5, 17)
    assert date_key("May 2020") == (2020, 5)
    assert date_key("sometime in 2020") is None and date_key("2020-13") is None
    assert is_before("2019", "2020-01") and not is_before("2020", "2020-05")
    assert not is_before("x", "2020")


def test_parse_reason():
    assert parse_update_classification("OPERATION: CORRECT\nREASON: it was wrong") == (
        "CORRECT", "it was wrong")


# ---- classifier_mode = "llm_only" ----

class Scripted:
    def __init__(self, *responses):
        self.responses, self.prompts = list(responses), []

    def get_completion(self, prompt, temperature=0.7):
        self.prompts.append(prompt)
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def test_llm_only_still_skips_llm_when_nothing_to_decide():
    store = Store()
    c = UpdateClassifier(NoLLM(), store, mode="llm_only")
    assert c.classify("s0", new(), 1) == ("ADD", [])            # empty key
    seed(store, "Lahore")
    assert c.classify("s0", new("lahore"), 2)[0] == "REAFFIRM"  # identical value


def test_llm_only_asks_even_where_rules_would_decide():
    store = Store()
    a = seed(store, "Lahore", 5, valid_from="2022")
    llm = Scripted("OPERATION: ADD_HISTORICAL\nREASON: earlier period")
    c = UpdateClassifier(llm, store, mode="llm_only")
    op, old = c.classify("s0", new("Paris", valid_from="2019"), 6)  # rules: ADD_HISTORICAL, no LLM
    assert op == "ADD_HISTORICAL" and [f.fact_id for f in old] == [a.fact_id]
    assert c.last_decision["source"] == "LLM" and len(llm.prompts) == 1
    assert "ADD_HISTORICAL" in llm.prompts[0]


def test_llm_only_ignores_learned_multivalued_rule():
    store = Store()
    seed(store, "Lahore", 1)
    seed(store, "Karachi", 2)
    llm = Scripted("OPERATION: REPLACE\nREASON: moved")
    c = UpdateClassifier(llm, store, mode="llm_only")
    op, old = c.classify("s0", new("Paris"), 3)
    assert op == "REPLACE" and len(old) == 2 and len(llm.prompts) == 1
    assert "Lahore" in llm.prompts[0] and "Karachi" in llm.prompts[0]


def test_llm_only_retries_once_on_unparseable():
    store = Store()
    seed(store)
    llm = Scripted("garbage", "OPERATION: CORRECT\nREASON: wrong")
    c = UpdateClassifier(llm, store, mode="llm_only")
    assert c.classify("s0", new(), 2)[0] == "CORRECT" and len(llm.prompts) == 2


def test_llm_only_falls_back_to_rules_after_two_failures():
    store = Store()
    seed(store, "Lahore", 5, valid_from="2022")
    llm = Scripted("garbage")
    c = UpdateClassifier(llm, store, mode="llm_only")
    op, _ = c.classify("s0", new("Paris", valid_from="2019"), 6)
    assert op == "ADD_HISTORICAL"                       # the date rule decided
    assert c.last_decision["source"] == "fallback:valid_from_earlier"
    assert len(llm.prompts) == 2


def test_llm_only_is_unresolved_when_no_rule_decides():
    store = Store()
    seed(store)
    c = UpdateClassifier(Canned("garbage"), store, mode="llm_only")
    assert c.classify("s0", new(), 2)[0] == "UNRESOLVED"
    assert c.last_decision["source"] == "fallback:LLM"


def test_parse_add_historical_only_when_allowed():
    allowed = ("REPLACE", "CORRECT", "ADD", "ADD_HISTORICAL")
    assert parse_update_classification("OPERATION: ADD_HISTORICAL\nREASON: x", allowed)[0] == "ADD_HISTORICAL"
    assert parse_update_classification("OPERATION: ADD\nREASON: x", allowed)[0] == "ADD"
    assert parse_update_classification("OPERATION: ADD_HISTORICAL\nREASON: x")[0] is None  # not allowed here
    assert parse_update_classification("nonsense", allowed) == (None, "")


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        UpdateClassifier(NoLLM(), Store(), mode="nope")


# ---- classifier_mode = "newest_wins" ----

def test_newest_wins_replaces_a_differing_value_without_the_llm():
    store = Store()
    old = seed(store)
    c = UpdateClassifier(NoLLM(), store, mode="newest_wins")
    assert c.classify("s0", new(), 5) == ("REPLACE", [old])
    assert c.last_decision["source"] == "newest_wins" and c.last_decision["raw"] is None


def test_newest_wins_still_adds_and_reaffirms():
    store = Store()
    c = UpdateClassifier(NoLLM(), store, mode="newest_wins")
    assert c.classify("s0", new(), 5) == ("ADD", [])
    old = seed(store, "Karachi")
    assert c.classify("s0", new(), 6) == ("REAFFIRM", [old])


def test_newest_wins_replaces_every_active_value():
    store = Store()
    old = [seed(store, "Lahore", 1), seed(store, "Dubai", 2)]
    c = UpdateClassifier(NoLLM(), store, mode="newest_wins")
    op, touched = c.classify("s0", new(), 5)
    assert op == "REPLACE" and [f.fact_id for f in touched] == [f.fact_id for f in old]


def test_newest_wins_ignores_additive_and_correction_wording():
    store = Store()
    seed(store)
    c = UpdateClassifier(NoLLM(), store, mode="newest_wins", seed_multivalued=True, additive_cue=True,
                         multivalued_attributes=["lives_in"])
    for text in ("She also lives in Karachi.", "Actually, she lives in Karachi."):
        assert c.classify("s0", new(source_text=text), 5)[0] == "REPLACE"


def test_newest_wins_ignores_a_past_tense_earlier_start():
    store = Store()
    seed(store, valid_from="2020")
    c = UpdateClassifier(NoLLM(), store, mode="newest_wins")
    assert c.classify("s0", new(valid_from="2018", temporal_status="PAST"), 5)[0] == "REPLACE"
