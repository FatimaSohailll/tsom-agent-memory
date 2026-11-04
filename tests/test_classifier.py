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
    assert [f.fact_id for f in old] == ([] if expected == "ADD" else [a.fact_id])
    p = llm.prompts[0]
    assert "Alice | lives_in | Lahore (learned at step 1)" in p
    assert "alice | Lives_In | Karachi (learned at step 2)" in p
    assert "She moved to Karachi." in p


@pytest.mark.parametrize("response", ["", "garbage", "OPERATION: maybe\nREASON: unsure",
                                      "REASON: it should CORRECT this"])
def test_unparseable_defaults_to_replace(response):
    store = Store()
    seed(store)
    op, old = UpdateClassifier(Canned(response), store).classify("s0", new(), 2)
    assert op == "REPLACE" and len(old) == 1


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
    op, _ = UpdateClassifier(llm, store).classify("s0", new("Paris", valid_from="2020"), 7)
    assert op == "REPLACE" and len(llm.prompts) == 1


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
