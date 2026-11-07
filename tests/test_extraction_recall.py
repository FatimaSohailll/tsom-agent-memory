"""Extraction recall on the change scenarios."""

import json

import pytest

from harness import eval_extraction
from harness.eval_extraction import evaluate_scenario, match, summarise
from harness.make_tables import make_tables
from harness.scenarios import Scenario, ScenarioFact, load_scenarios
from tsom.contracts import RunConfig
from tsom.llm import LLM
from tsom.temporal.extractor import FactExtractor


def expected(value="Lahore", speaker="Ali", session=0):
    return ScenarioFact(speaker=speaker, predicate="lives_in", value=value, session=session,
                        expected_label="additive")


def got(entity, value, attribute="lives_in"):
    return {"entity": entity, "attribute": attribute, "value": value}


@pytest.mark.parametrize("facts,outcome", [
    ([got("Ali", "Lahore")], "found"),
    ([got("ali", "LAHORE.")], "found"),
    ([got("Ali", "Lahore", "home_city")], "found"),  # attribute names are free-form
    ([got("Ali Khan", "Lahore")], "found"),
    ([got("Ali", "Lahore, Pakistan")], "found"),
    ([got("Ali", "city of Lahore")], "found"),
    ([got("I", "Lahore")], "wrong_entity"),
    ([got("Sara", "Lahore")], "wrong_entity"),
    ([got("Ali", "Karachi")], "missed"),
    ([got("Ali", "Lahor")], "missed"),
    ([], "missed"),
])
def test_match(facts, outcome):
    assert match(expected(), facts) == outcome


def test_a_longer_expected_value_is_found_from_its_main_word():
    assert match(expected("Meezan Bank"), [got("Ali", "Meezan")]) == "found"


def test_a_word_inside_another_word_does_not_match():
    assert match(expected("tea"), [got("Ali", "steak")]) == "missed"


def test_the_right_entity_wins_over_a_wrong_one():
    assert match(expected(), [got("Sara", "Lahore"), got("Ali", "Lahore")]) == "found"


def block(entity, value):
    return f"FACT 1:\nENTITY: {entity}\nATTRIBUTE: lives_in\nVALUE: {value}\nVALID_FROM: NONE\nVALID_UNTIL: NONE"


class Scripted(LLM):
    """Replies by a word in the turn; unmatched turns have no facts."""

    def __init__(self, replies):
        super().__init__("fake", "fake")
        self.replies, self.prompts = replies, []

    def _call(self, prompt, temperature):
        self.prompts.append(prompt)
        text = prompt.split("Text:")[-1]
        for word, reply in self.replies.items():
            if word in text:
                return reply, 1, 1
        return "NO_FACTS", 1, 1


def scenario(**over):
    data = dict(
        id="t-1", kind="change", reviewed=True,
        sessions=[{"date": "1:56 pm on 8 May, 2023", "turns": [{"speaker": "Ali", "text": "I live in Karachi."}]},
                  {"date": "6:30 pm on 14 June, 2023", "turns": [{"speaker": "Ali", "text": "I moved to Lahore."}]}],
        facts=[{"speaker": "Ali", "predicate": "lives_in", "value": "Karachi", "session": 0, "expected_label": "additive"},
               {"speaker": "Ali", "predicate": "lives_in", "value": "Lahore", "session": 1,
                "expected_label": "corrective", "replaces": "Karachi"}],
        question="Where does Ali live now?", question_alt="Which city is Ali based in?",
        current=["lahore"], old=["karachi"])
    data.update(over)
    return Scenario(**data)


def test_each_fact_is_looked_for_in_its_own_session():
    llm = Scripted({"Karachi": block("Ali", "Lahore"), "Lahore": block("Ali", "Karachi")})
    result = evaluate_scenario(FactExtractor(llm), scenario())
    assert [f["outcome"] for f in result["facts"]] == ["missed", "missed"]


def test_the_extractor_is_told_who_is_speaking():
    llm = Scripted({})
    evaluate_scenario(FactExtractor(llm), scenario())
    assert all("Ali" in prompt for prompt in llm.prompts)


def test_summary_counts_outcomes_by_kind():
    llm = Scripted({"Karachi": block("Ali", "Karachi"), "Lahore": block("I", "Lahore")})
    summary = summarise([evaluate_scenario(FactExtractor(llm), scenario())])
    assert summary["overall"]["recall"] == {"k": 1, "n": 2, "pct": 50.0}
    assert summary["overall"]["wrong_entity"] == 1 and summary["overall"]["missed"] == 0
    assert summary["per_kind"]["change"]["recall"]["k"] == 1
    assert [f["found"] for f in summary["per_fact"]] == [True, False]


def fake_config(tmp_path, **fields):
    return RunConfig(run_name="ext", system="tsom", backend="fake", model="fake",
                     output_dir=str(tmp_path), **fields)


def test_run_covers_every_reviewed_fact_and_writes_its_files(tmp_path):
    summary = eval_extraction.run(fake_config(tmp_path))
    assert summary["overall"]["recall"]["n"] == sum(len(s.facts) for s in load_scenarios())
    assert summary["reviewed_only"] is True
    assert (tmp_path / "ext" / "extraction_summary.md").exists()
    rows = (tmp_path / "ext" / "extraction_results.jsonl").read_text().splitlines()
    assert len(rows) == len(load_scenarios())


def test_unreviewed_labels_are_marked(tmp_path):
    summary = eval_extraction.run(fake_config(tmp_path), "data/scenarios/drafts.jsonl",
                                  include_unreviewed=True)
    assert summary["reviewed_only"] is False
    assert "unreviewed" in (tmp_path / "ext" / "extraction_summary.md").read_text()


def test_tables_show_extraction_recall(tmp_path):
    folder = tmp_path / "t"
    folder.mkdir()
    summary = {"per_fact": [{"found": True}, {"found": False}]}
    (folder / "extraction_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    text = make_tables({"tsom": [folder]})
    assert "## Extraction recall" in text and "50.0%" in text


def test_tables_say_when_extraction_is_missing(tmp_path):
    assert "No extraction results found." in make_tables({"tsom": [tmp_path / "none"]})
