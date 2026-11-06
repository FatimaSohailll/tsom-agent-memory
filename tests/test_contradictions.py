"""Active contradiction rate: rebuilt from the fact table at each session checkpoint."""

import pytest

from harness.contradictions import contradiction_line, curve, read_curve, summarise_curves
from harness.run_factcons import load_samples, run as run_factcons
from harness.run_locomo import run as run_locomo
from tsom.contracts import Fact, RunConfig
from tsom.store import Store


def add(store, value, step, session, attribute="lives_in", **fields):
    fact = Fact(sample_id="s", note_id="n", entity="Ali", attribute=attribute, value=value,
                observed_at=step, observed_time=session, **fields)
    store.add_fact(fact)
    return fact


def replace(store, old, value, step, session, status="superseded"):
    new = add(store, value, step, session, supersedes=[old.fact_id])
    store.update_fact_status(old.fact_id, status, superseded_by=new.fact_id)
    return new


def rates(store, multivalued=()):
    return [(p["facts"], p["contradictions"], p["all_facts"], p["without_consolidation"])
            for p in curve(store, "s", multivalued)]


def test_a_replaced_value_is_no_contradiction_but_would_be_without_consolidation():
    store = Store()
    replace(store, add(store, "Lahore", 1, "t1"), "Karachi", 2, "t2")
    assert rates(store) == [(1, 0, 1, 0), (1, 0, 2, 1)]


def test_a_wrong_add_stays_a_contradiction():
    store = Store()
    add(store, "Lahore", 1, "t1")
    add(store, "Karachi", 2, "t2")
    assert rates(store) == [(1, 0, 1, 0), (2, 1, 2, 1)]


def test_a_contradiction_lasts_until_the_step_that_resolves_it():
    store = Store()
    first, second = add(store, "Lahore", 1, "t1"), add(store, "Karachi", 2, "t2")
    third = add(store, "Dubai", 3, "t3", supersedes=[first.fact_id, second.fact_id])
    for old in (first, second):
        store.update_fact_status(old.fact_id, "superseded", superseded_by=third.fact_id)
    assert [p["contradictions"] for p in curve(store, "s", ())] == [0, 1, 0]


def test_a_retracted_value_counts_only_without_consolidation():
    store = Store()
    replace(store, add(store, "Lahore", 1, "t1"), "Karachi", 2, "t2", status="retracted")
    assert rates(store)[-1] == (1, 0, 2, 1)


def test_a_late_historical_fact_is_never_active():
    store = Store()
    current = add(store, "Karachi", 1, "t1")
    add(store, "Dubai", 2, "t2", status="superseded", superseded_by=current.fact_id)
    assert rates(store)[-1] == (1, 0, 2, 1)


def test_multi_valued_attributes_are_not_contradictions():
    store = Store()
    add(store, "Urdu", 1, "t1", attribute="speaks")
    add(store, "English", 2, "t1", attribute="speaks")
    assert rates(store, ("speaks",)) == [(2, 0, 2, 0)]
    assert rates(store) == [(2, 1, 2, 1)]


def test_the_same_value_in_other_spelling_is_not_a_contradiction():
    store = Store()
    add(store, "Lahore", 1, "t1")
    add(store, "lahore ", 2, "t1")
    assert rates(store)[-1][1] == 0


def test_facts_of_one_session_share_a_checkpoint():
    store = Store()
    add(store, "a", 1, "t1", attribute="x")
    add(store, "b", 2, "t1", attribute="y")
    add(store, "c", 3, "t2", attribute="z")
    assert [p["session"] for p in curve(store, "s", ())] == ["t1", "t2"]


def test_rates_are_per_100_facts():
    store = Store()
    add(store, "Lahore", 1, "t1")
    add(store, "Karachi", 2, "t1")
    add(store, "Ahmed", 3, "t1", attribute="friend")
    add(store, "Sara", 4, "t1", attribute="sister")
    point = curve(store, "s", ())[0]
    assert point["rate"] == 25.0 and point["rate_without_consolidation"] == 25.0


def test_summary_compares_each_checkpoint():
    store = Store()
    add(store, "Lahore", 1, "t1")
    replace(store, store.active_facts("s", "Ali", "lives_in")[0], "Karachi", 2, "t2")
    summary = summarise_curves({"s": curve(store, "s", ())})
    assert summary["checkpoints"] == 2 and summary["checkpoints_lower"] == 1
    assert summary["final_rate"] == 0.0 and summary["final_rate_without_consolidation"] == 50.0
    assert "1 of 2" in contradiction_line(summary)


def test_no_facts_give_an_empty_summary():
    assert summarise_curves({}) is None and contradiction_line(None) == ""


def test_a_missing_database_gives_no_curve(tmp_path):
    assert read_curve(tmp_path / "none.sqlite", "s", ()) == []


ROWS = [{"context": "Here is a list of facts:\n0. Alice is Lahore.\n1. Alice is Karachi.\n",
         "questions": ["Where is Alice?"], "answers": [["Karachi"]],
         "metadata": {"source": "factconsolidation_sh_6k"}}]


def factcons(system, tmp_path):
    config = RunConfig(run_name=system, system=system, backend="fake", model="fake",
                       output_dir=str(tmp_path))
    samples = load_samples("sh", "6k", loader=lambda: ROWS)
    return lambda: run_factcons(config, "sh", "6k", chunk=1, samples=samples)


def test_factconsolidation_summary_reports_contradictions_and_keeps_them_on_resume(tmp_path):
    start = factcons("tsom", tmp_path)
    summary = start()
    assert summary["contradictions"]["checkpoints"] == 2
    assert "per 100 facts" in (tmp_path / "tsom" / "summary_factcons_sh_6k.md").read_text()
    assert start() == summary


@pytest.mark.parametrize("system", ["amem", "rag"])
def test_systems_without_facts_report_no_contradictions(system, tmp_path):
    summary = factcons(system, tmp_path)()
    assert summary["contradictions"] is None
    assert "per 100 facts" not in (tmp_path / system / "summary_factcons_sh_6k.md").read_text()


def test_locomo_summary_has_a_checkpoint_per_session_with_facts(tmp_path):
    config = RunConfig(run_name="smoke", system="tsom", backend="fake", model="fake", max_samples=1,
                       max_sessions=2, max_questions=1, output_dir=str(tmp_path))
    summary = run_locomo(config)
    assert set(summary["contradictions"]) >= {"checkpoints", "final_rate", "per_sample"}
    assert set(summary["contradictions"]["per_sample"]) == {"0"}
