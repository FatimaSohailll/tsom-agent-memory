"""Change scenarios: loader, generator, review, scorer and runner."""

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from harness import run_scenarios
from harness.scenarios import (DEFAULT_SCENARIOS, Scenario, load_scenarios, read_jsonl, score_answer,
                               write_jsonl)
from harness.scenarios import answered as subset
from tsom.contracts import RunConfig
from tsom.llm import LLM

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
    with pytest.raises(ValidationError):
        scenario(kind="restate")  # restate has no old answers
    with pytest.raises(ValidationError):
        scenario(sessions=scenario().sessions[:1])  # at least 2 sessions


def test_hand_written_scenarios_are_reviewed():
    hand = load_scenarios(DEFAULT_SCENARIOS)
    assert len(hand) >= 6
    assert {"change", "additive", "restate", "conflict"} <= {s.kind for s in hand}


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


def test_the_original_drafts_never_change():
    original = [d.model_dump() for d in generate_scenarios.generate()]
    assert [d.model_dump() for d in generate_scenarios.generate_all()[:len(original)]] == original


def test_the_extra_batch_adds_cancellations_and_unique_ids():
    drafts = generate_scenarios.generate_all()
    kinds = [d.kind for d in drafts]
    expected = {k: generate_scenarios.COUNTS.get(k, 0) + generate_scenarios.EXTRA_COUNTS.get(k, 0)
                for k in set(generate_scenarios.COUNTS) | set(generate_scenarios.EXTRA_COUNTS)}
    assert {k: kinds.count(k) for k in set(kinds)} == expected
    assert len({d.id for d in drafts}) == len(drafts)
    assert len({(d.facts[0].speaker, d.facts[0].value, d.facts[-1].value, d.question) for d in drafts}) == len(drafts)
    for d in drafts:
        for f in d.facts:
            text = " ".join(t.text for t in d.sessions[f.session].turns).lower()
            assert f.value.lower() in text, (d.id, f.value)


def test_a_cancelled_booking_is_scored_as_current_or_stale():
    cancel = next(d for d in generate_scenarios.generate_all() if d.kind == "cancel")
    booking = cancel.facts[0].value
    assert score_answer(f"The {booking} was cancelled.", cancel)["verdict"] == "correct"
    assert score_answer(f"The {booking} is confirmed for Friday.", cancel)["verdict"] == "stale"
    assert score_answer("It was called off.", cancel)["correct"]


def test_regenerating_keeps_hand_edits_and_reviewed_marks():
    edited = generate_scenarios.generate()[0].model_copy(update={"question": "edited?", "reviewed": True})
    merged = generate_scenarios.merge([edited], generate_scenarios.generate_all())
    assert merged[0].question == "edited?" and merged[0].reviewed
    assert len(merged) == len(generate_scenarios.generate_all())


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


# ---- runner ----

def fake_config(tmp_path, name, **flags):
    return RunConfig(run_name=name, backend="fake", model="fake", output_dir=str(tmp_path), **flags)


def answered():
    return [s for s in load_scenarios(DEFAULT_SCENARIOS) if s.kind != "conflict"]


def test_run_scenarios_smoke_and_resume(tmp_path):
    config = fake_config(tmp_path, "tsom", system="tsom")
    summary = run_scenarios.run(config)
    out = tmp_path / "tsom"
    assert summary["overall"]["scenarios"] == len(answered())  # conflict scenarios are skipped
    assert summary["cost"]["writes"] > 0 and summary["cost"]["llm_calls_per_write"] > 0
    rows = [json.loads(line) for line in (out / "scenarios_results.jsonl").read_text().splitlines()]
    # two wordings of the question, plus the past-period question where the scenario has one
    assert all(len(r["answers"]) == (3 if r["past_correct"] is not None else 2) for r in rows)
    assert all("context" in r["answers"][0] for r in rows) and any(r["past_correct"] is not None for r in rows)
    assert run_scenarios.run(config) == summary


@pytest.mark.parametrize("system", ["amem", "rag"])
def test_run_scenarios_covers_the_baselines(system, tmp_path):
    assert run_scenarios.run(fake_config(tmp_path, system, system=system), limit=2)["overall"]["scenarios"] == 2


def test_run_scenarios_defaults_to_temperature_zero(tmp_path):
    summary = run_scenarios.run(fake_config(tmp_path, "t", system="rag"), limit=1)
    assert summary["temperature"] == 0.0
    assert json.loads((tmp_path / "t" / "config.json").read_text())["temperature"] == 0.0


def test_run_scenarios_keeps_a_temperature_the_config_sets(tmp_path):
    summary = run_scenarios.run(fake_config(tmp_path, "t", system="rag", temperature=0.1), limit=1)
    assert summary["temperature"] == 0.1


def test_the_fact_layer_marks_the_new_value_current(tmp_path):
    sc = scenario(sessions=[{"date": "1:56 pm on 8 May, 2023", "turns": [{"speaker": "Ali", "text": "Ali is Karachi."}]},
                            {"date": "6:30 pm on 14 June, 2023", "turns": [{"speaker": "Ali", "text": "Ali is Lahore."}]}],
                  question="Where is Ali?")
    config = fake_config(tmp_path, "t", system="tsom")
    result = run_scenarios.run_scenario(config, LLM("fake", "fake"), sc)
    assert "[CURRENT] Ali | is | Lahore" in result["answers"][0]["context"]


def test_a_limit_is_shared_equally_across_kinds():
    kinds = ["change"] * 6 + ["cancel"] * 3 + ["additive"] * 2 + ["conflict"] * 4
    pool = [scenario(id=f"s{i}", kind=k, old=["karachi"] if k in ("change", "cancel") else [])
            for i, k in enumerate(kinds)]
    picked = subset(pool, 7)
    counts = {k: sum(s.kind == k for s in picked) for k in ("change", "cancel", "additive", "conflict")}
    assert counts == {"change": 3, "cancel": 2, "additive": 2, "conflict": 0}
    assert [s.id for s in picked] == sorted((s.id for s in picked), key=lambda i: int(i[1:]))
    assert len(subset(pool)) == 11 and subset(pool, 100) == subset(pool)


def test_extraction_and_answering_use_the_same_limited_scenarios(tmp_path):
    from harness import eval_extraction
    config = RunConfig(run_name="e", system="tsom", backend="fake", model="fake", output_dir=str(tmp_path))
    ids = [json.loads(line)["id"] for line in (lambda s: (eval_extraction.run(config, limit=4),
           (tmp_path / "e" / "extraction_results.jsonl").read_text().splitlines())[1])(None)]
    assert ids == [s.id for s in subset(load_scenarios(), 4)]

