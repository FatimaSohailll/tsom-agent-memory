"""Change scenarios: loader, generator, review, scorer, and both runners."""

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from harness import eval_actions, run_scenarios
from harness.scenarios import DEFAULT_SCENARIOS, Scenario, load_scenarios, read_jsonl, score_answer, write_jsonl
from tsom.contracts import RunConfig
from tsom.llm import LLM

from scripted import ScriptedLLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import generate_scenarios  # noqa: E402
import review_scenarios  # noqa: E402


def scenario(**over):
    data = dict(
        id="t-1", kind="change", reviewed=True,
        sessions=[{"date": "1:56 pm on 8 May, 2023",
                   "turns": [{"speaker": "Ali", "text": "I live in Karachi and I like tea."}]},
                  {"date": "6:30 pm on 14 June, 2023",
                   "turns": [{"speaker": "Ali", "text": "I moved to Lahore. I also like coffee."}]}],
        facts=[{"speaker": "Ali", "predicate": "lives_in", "value": "Karachi", "session": 0, "expected_label": "additive"},
               {"speaker": "Ali", "predicate": "likes", "value": "tea", "session": 0, "expected_label": "additive"},
               {"speaker": "Ali", "predicate": "lives_in", "value": "Lahore", "session": 1,
                "expected_label": "corrective", "replaces": "Karachi"},
               {"speaker": "Ali", "predicate": "likes", "value": "coffee", "session": 1, "expected_label": "additive"}],
        question="Where does Ali live now?", question_alt="Which city is Ali based in?",
        current=["lahore"], old=["karachi"])
    data.update(over)
    return Scenario(**data)


# ---- data and loader ----

def test_loader_skips_unreviewed(tmp_path):
    path = tmp_path / "s.jsonl"
    write_jsonl(path, [scenario(id="a"), scenario(id="b", reviewed=False)])
    assert [s.id for s in load_scenarios(path)] == ["a"]
    assert [s.id for s in load_scenarios(path, include_unreviewed=True)] == ["a", "b"]


def test_validation():
    with pytest.raises(ValidationError, match="unknown predicate"):
        scenario(facts=[{"speaker": "Ali", "predicate": "favourite_colour", "value": "blue",
                         "session": 0, "expected_label": "additive"}])
    with pytest.raises(ValidationError):
        scenario(kind="restate")  # restate has no old answers
    with pytest.raises(ValidationError):
        scenario(sessions=scenario().sessions[:1])  # at least 2 sessions


def test_hand_written_scenarios_are_reviewed():
    hand = load_scenarios(DEFAULT_SCENARIOS)
    assert len(hand) >= 6
    assert {s.kind for s in hand} == {"change", "additive", "restate", "conflict"}


def test_generator_counts_and_determinism():
    assert len(generate_scenarios.TEMPLATES) >= 12
    assert len({t["predicate"] for t in generate_scenarios.TEMPLATES}) >= 8
    drafts = generate_scenarios.generate()
    kinds = [d.kind for d in drafts]
    assert {k: kinds.count(k) for k in set(kinds)} == {"change": 21, "additive": 7, "restate": 3, "conflict": 3}
    assert not any(d.reviewed for d in drafts)
    assert len({d.id for d in drafts}) == len(drafts)
    assert all(2 <= len(d.sessions) <= 4 for d in drafts)
    assert [d.model_dump() for d in drafts] == [d.model_dump() for d in generate_scenarios.generate()]
    for d in drafts:  # each later value appears in the text of its session
        for f in d.facts:
            text = " ".join(t.text for t in d.sessions[f.session].turns).lower()
            assert f.value.lower() in text, (d.id, f.value)


def test_paraphrase_never_drops_values():
    class Rewriter(LLM):
        def __init__(self):
            super().__init__("fake", "rewriter")

        def _call(self, prompt, temperature):
            return "Something completely different.", 1, 1
    drafts = generate_scenarios.generate({"change": 1, "additive": 0, "restate": 0, "conflict": 0})
    before = [t.text for s in drafts[0].sessions for t in s.turns]
    generate_scenarios.paraphrase(drafts, Rewriter())
    after = [t.text for s in drafts[0].sessions for t in s.turns]
    for b, a in zip(before, after):
        has_value = any(f.value.lower() in b.lower() for f in drafts[0].facts)
        assert a == b if has_value else a == "Something completely different."


def test_review_accepts_and_marks(tmp_path):
    drafts, out = tmp_path / "drafts.jsonl", tmp_path / "out.jsonl"
    write_jsonl(drafts, [scenario(id="d1", reviewed=False), scenario(id="d2", reviewed=False),
                         scenario(id="d3", reviewed=False)])
    answers = iter(["a", "s", "q"])
    review_scenarios.review(drafts, out, input_fn=lambda _: next(answers), print_fn=lambda *_: None)
    assert [(s.id, s.reviewed) for s in read_jsonl(out)] == [("d1", True)]
    assert [s.reviewed for s in read_jsonl(drafts)] == [True, False, False]


# ---- answer scorer ----

def test_score_change():
    sc = scenario()
    assert score_answer("Ali lives in Lahore.", sc)["verdict"] == "correct"
    assert score_answer("KARACHI!", sc) == {"correct": False, "stale": True, "hedged": False,
                                            "verdict": "stale", "retention": None}
    both = score_answer("He moved from Karachi to Lahore", sc)
    assert both["correct"] and both["hedged"] and not both["stale"] and both["verdict"] == "hedged"
    assert score_answer("Not mentioned", sc)["verdict"] == "wrong"


def test_score_additive_needs_all_values_and_retention():
    sc = scenario(kind="additive", old=[], current=["dog", "parrot"],
                  facts=[{"speaker": "Ali", "predicate": "has_pet", "value": "dog", "session": 0, "expected_label": "additive"},
                         {"speaker": "Ali", "predicate": "has_pet", "value": "parrot", "session": 1, "expected_label": "additive"}])
    assert score_answer("A dog, and a parrot.", sc)["correct"]
    only_new = score_answer("a parrot", sc)
    assert not only_new["correct"] and only_new["retention"] is False
    assert score_answer("just the dog", sc)["retention"] is True


def test_score_restate_retention():
    sc = scenario(kind="restate", old=[], current=["teacher"],
                  facts=[{"speaker": "Ali", "predicate": "job_title", "value": "teacher", "session": 0, "expected_label": "additive"},
                         {"speaker": "Ali", "predicate": "job_title", "value": "teacher", "session": 1, "expected_label": "redundant"}])
    s = score_answer("He is a teacher.", sc)
    assert s["correct"] and s["retention"]


# ---- eval_actions ----

def test_eval_actions_known_outcomes():
    llm = ScriptedLLM([
        "FACT: lives_in | Karachi | additive | none\nFACT: likes | tea | additive | none",
        # Lahore labelled additive -> applied as contradictory, Karachi stays active (missed correction);
        # coffee labelled corrective against tea -> tea retired (false retirement).
        "FACT: lives_in | Lahore | additive | none\nFACT: likes | coffee | corrective | tea",
    ])
    r = eval_actions.evaluate_scenario(scenario(), llm, use_classifier=True)
    assert r["false_retirements"] == {"k": 1, "n": 3, "items": ["Ali likes tea"]}
    assert r["missed_corrections"] == {"k": 1, "n": 1, "items": ["Ali lives_in Karachi"]}
    assert r["turns"] == 2 and r["llm_calls"] == 2 and r["parse_failures"] == 0 and r["spurious"] == 0
    s = eval_actions.summarise_actions([r])["overall"]
    assert s["extraction_recall"] == {"k": 4, "n": 4, "pct": 100.0}
    assert s["label_accuracy"] == {"k": 2, "n": 4, "pct": 50.0}
    assert s["confusion"]["corrective"]["additive"] == 1
    assert s["confusion"]["additive"]["corrective"] == 1
    assert s["confusion"]["additive"]["additive"] == 2
    lahore = next(f for f in r["facts"] if f["expected"]["value"] == "Lahore")
    assert lahore["got_label"] == "additive" and lahore["applied_label"] == "contradictory"
    assert lahore["raw_line"] == "FACT: lives_in | Lahore | additive | none"


def test_eval_actions_missed_extraction_and_parse_failure():
    llm = ScriptedLLM(["garbage", "still garbage", "FACT: lives_in | lahore city | corrective | Karachi"])
    r = eval_actions.evaluate_scenario(scenario(), llm, use_classifier=False)
    assert [f["found"] for f in r["facts"]] == [False, False, True, False]
    assert r["parse_failures"] == 1 and r["llm_calls"] == 3
    assert r["missed_corrections"]["n"] == 0  # Karachi was never stored


def fake_config(tmp_path, name, **flags):
    return RunConfig(run_name=name, backend="fake", model="fake", output_dir=str(tmp_path), **flags)


def test_eval_actions_smoke_and_resume(tmp_path):
    config = fake_config(tmp_path, "gap1", system="tsom", use_classifier=True)
    summary = eval_actions.run(config)
    out = tmp_path / "gap1"
    assert summary["overall"]["scenarios"] == 6
    assert (out / "actions_summary.md").exists()
    facts = [json.loads(line) for line in (out / "actions_facts.jsonl").read_text().splitlines()]
    assert len(facts) == summary["overall"]["extraction_recall"]["n"] and {"scenario", "expected"} <= set(facts[0])
    assert eval_actions.run(config) == summary  # resumed from the per-scenario cache


def test_run_scenarios_smoke_and_resume(tmp_path):
    config = fake_config(tmp_path, "both", system="tsom", use_classifier=True, use_validity=True)
    summary = run_scenarios.run(config)
    out = tmp_path / "both"
    assert summary["overall"]["scenarios"] == 5  # the conflict scenario is skipped
    assert summary["cost"]["writes"] > 0 and summary["cost"]["llm_calls_per_write"] >= 2
    rows = [json.loads(line) for line in (out / "scenarios_results.jsonl").read_text().splitlines()]
    assert all(len(r["answers"]) == 2 and "context" in r["answers"][0] for r in rows)
    assert run_scenarios.run(config) == summary
    amem = run_scenarios.run(fake_config(tmp_path, "amem"), limit=2)
    assert amem["overall"]["scenarios"] == 2
