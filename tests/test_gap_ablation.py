"""Gap-wise ablation: Gap 1 off (append-only classifier), Gap 3 off (no time-scoped retrieval),
and the past-period questions that measure Gap 3."""

import json
from pathlib import Path

import pytest

from harness import run_scenarios
from harness.make_tables import make_tables
from harness.scenarios import Scenario, past_question, score_past, with_past_question
from tsom.contracts import Fact, RunConfig
from tsom.llm import LLM
from tsom.memory import MemorySystem
from tsom.store import Store
from tsom.temporal.classifier import UpdateClassifier

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


class NoLLM:
    def get_completion(self, prompt, temperature=0.7):
        raise AssertionError("no LLM call expected")


def stored(store, value, step=1):
    fact = Fact(sample_id="s", note_id="n", entity="Ali", attribute="lives_in", value=value, observed_at=step)
    store.add_fact(fact)
    return fact


def test_append_only_adds_every_new_value_without_the_llm():
    store = Store()
    old = stored(store, "Karachi")
    c = UpdateClassifier(NoLLM(), store, mode="append_only")
    new = {"entity": "Ali", "attribute": "lives_in", "value": "Lahore"}
    assert c.classify("s", new, 2) == ("ADD", [old])
    assert c.last_decision["source"] == "append_only"
    assert c.classify("s", {**new, "value": "Karachi"}, 3)[0] == "REAFFIRM"


def test_append_only_keeps_both_values_current():
    memory = MemorySystem(RunConfig(run_name="t", system="tsom", backend="fake", model="fake",
                                    classifier_mode="append_only"), LLM("fake", "fake"), sample_id="s0")
    memory.write("Alice is Lahore.", "t1")
    memory.write("Alice is Karachi.", "t2")
    assert [f.status for f in memory.store.fact_history("s0", "Alice", "is")] == ["active", "active"]


class Scripted(LLM):
    def __init__(self, values):
        super().__init__("fake", "fake")
        self.values = list(values)

    def get_completion(self, prompt, temperature=0.7):
        if self.values and prompt.startswith("Extract the atomic facts stated in the text."):
            self.usage.calls += 1
            return (f"FACT 1:\nENTITY: Alice\nATTRIBUTE: lives_in\nVALUE: {self.values.pop(0)}\n"
                    "VALID_FROM: NONE\nVALID_UNTIL: NONE")
        return super().get_completion(prompt, temperature)


def moved(**fields):
    memory = MemorySystem(RunConfig(run_name="t", system="tsom", backend="fake", model="fake", **fields),
                          Scripted(["Karachi", "Lahore"]), sample_id="s0")
    memory.write("I live in Karachi.", "1:56 pm on 8 January, 2023")
    memory.write("I moved to Lahore.", "6:30 pm on 14 June, 2023")
    return memory


def shown(memory, question):
    return [line.split(" | ")[2].split(" (")[0] for line in memory.search(question, 5)["facts"].splitlines()]


def test_time_scoped_retrieval_answers_by_period():
    assert shown(moved(), "where did Alice live in March 2023") == ["Karachi"]


def test_without_time_scoping_every_question_gets_the_current_facts():
    memory = moved(time_scoped_retrieval=False)
    for question in ("where did Alice live in March 2023", "where did Alice live before", "where does Alice live"):
        assert shown(memory, question) == ["Lahore"]
        assert memory.search(question, 5)["query_type"] == "current"
    assert all(f.status in ("active", "superseded") for f in memory.store.fact_history("s0", "Alice", "lives_in"))


def test_gap_three_alone_has_no_replacements_to_scope():
    memory = moved(classifier_mode="append_only")
    assert shown(memory, "where did Alice live in March 2023") == ["Karachi", "Lahore"]


@pytest.mark.parametrize("name,field,value", [("tsom_gap1only_qwen2.5-3b", "time_scoped_retrieval", False),
                                              ("tsom_gap3only_qwen2.5-3b", "classifier_mode", "append_only")])
def test_ablation_configs(name, field, value):
    config = RunConfig(**json.loads((CONFIGS / f"{name}.json").read_text()))
    assert getattr(config, field) == value and config.system == "tsom"


def scenario(kind="change", first="1:56 pm on 8 January, 2023", later="6:30 pm on 14 June, 2023",
             predicate="lives_in", **over):
    old = {"change": ["karachi"], "cancel": ["still booked"]}.get(kind, [])
    data = dict(id="p-1", kind=kind, reviewed=True,
                sessions=[{"date": first, "turns": [{"speaker": "Ali", "text": "I live in Karachi."}]},
                          {"date": later, "turns": [{"speaker": "Ali", "text": "I moved to Lahore."}]}],
                facts=[{"speaker": "Ali", "predicate": predicate, "value": "Karachi", "session": 0,
                        "expected_label": "additive"},
                       {"speaker": "Ali", "predicate": predicate, "value": "Lahore", "session": 1,
                        "expected_label": "corrective", "replaces": "Karachi" if kind in ("change", "cancel") else None}],
                question="Where does Ali live now?", question_alt="Which city is Ali in?",
                current=["lahore"], old=old)
    data.update(over)
    return Scenario(**data)


def test_a_past_question_asks_about_the_first_values_month():
    assert past_question(scenario()) == ("Where did Ali live in January 2023?", ["karachi"])


def test_a_cancelled_booking_asks_what_was_booked():
    sc = scenario("cancel", predicate="has_booking", current=["cancelled"])
    assert past_question(sc) == ("What did Ali have booked in January 2023?", ["karachi"])


def test_values_a_month_apart_get_an_undated_question():
    # "I moved last month" could make a single month ambiguous, so no month is named
    sc = scenario(later="6:30 pm on 14 February, 2023")
    assert past_question(sc) == ("Where did Ali live before?", ["karachi"])


@pytest.mark.parametrize("sc", [
    scenario(predicate="age"),                        # no sensible past-period wording
    scenario(kind="additive", current=["karachi", "lahore"]),
])
def test_no_past_question_when_it_would_be_unfair(sc):
    assert past_question(sc) is None and with_past_question(sc).question_past is None


def test_scoring_a_past_answer():
    sc = with_past_question(scenario())
    assert score_past("He lived in Karachi.", sc) == {"past_correct": True, "past_gave_current": False}
    assert score_past("Lahore", sc) == {"past_correct": False, "past_gave_current": True}
    assert score_past("No idea", sc) == {"past_correct": False, "past_gave_current": False}


def test_an_existing_past_question_is_kept():
    sc = scenario(question_past="Where was Ali in winter?", past=["karachi"])
    assert with_past_question(sc).question_past == "Where was Ali in winter?"


def test_the_past_question_and_answers_go_together():
    with pytest.raises(ValueError):
        scenario(question_past="Where was Ali?")


def test_runs_ask_and_score_the_past_question(tmp_path):
    sc = with_past_question(scenario())
    config = RunConfig(run_name="t", system="tsom", backend="fake", model="fake")
    result = run_scenarios.run_scenario(config, LLM("fake", "fake"), sc)
    assert len(result["answers"]) == 3 and result["answers"][2]["past"] is True
    assert result["past_correct"] is False and result["past_gave_current"] is False
    summary = run_scenarios.summarise_scenarios([result])
    assert summary["overall"]["past_correct"] == {"k": 0, "n": 1, "pct": 0.0}


def test_runs_without_a_past_question_leave_it_unscored():
    config = RunConfig(run_name="t", system="rag", backend="fake", model="fake")
    result = run_scenarios.run_scenario(config, LLM("fake", "fake"), scenario())
    assert len(result["answers"]) == 2 and result["past_correct"] is None
    assert run_scenarios.summarise_scenarios([result])["overall"]["past_correct"]["n"] == 0


def test_tables_show_past_period_accuracy(tmp_path):
    folder = tmp_path / "t"
    folder.mkdir()
    rows = [{"correct": True, "stale": False, "hedged": False, "past_correct": p} for p in (True, False, None)]
    (folder / "scenarios_summary.json").write_text(json.dumps({"per_scenario": rows}), encoding="utf-8")
    text = make_tables({"tsom": [folder]})
    assert "Past-period correct" in text and "50.0% [" in text and "(n=2)" in text
