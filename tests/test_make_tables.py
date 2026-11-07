"""Result tables: confidence intervals, seeds, comparisons and missing runs."""

import argparse
import json

import pytest

from harness.make_tables import (SMALL_N, bootstrap_ci, comparison, make_tables, parse_run,
                                 rate_cell)


def test_bootstrap_is_repeatable_and_brackets_the_mean():
    values = [True] * 30 + [False] * 70
    low, high = bootstrap_ci(values)
    assert low < 30 < high and (low, high) == bootstrap_ci(values)
    assert bootstrap_ci([]) is None
    assert bootstrap_ci([True] * 5) == (100.0, 100.0)


def test_one_seed_shows_no_spread():
    text, _, n = rate_cell([[True, False, False, False]])
    assert text.startswith("25.0% [") and "±" not in text and n == 4


def test_seeds_show_their_spread_and_average_each_item():
    text, ci, n = rate_cell([[True, True, False, False], [True, False, False, False]])
    assert text.startswith("37.5 ± 17.7%") and n == 4
    assert ci is not None


def test_seeds_with_different_questions_are_refused():
    with pytest.raises(ValueError):
        rate_cell([[True, False], [True]])


def test_no_results_are_missing_not_zero():
    assert rate_cell([[], []]) == ("missing", None, 0)


def test_comparison_reports_separation():
    cis = {"amem": (40.0, 60.0), "tsom": (10.0, 30.0)}
    assert comparison("Stale", cis, {"amem": 50, "tsom": 50}) == [
        "Stale, tsom vs amem: lower, and the intervals do not overlap."]


def test_comparison_flags_small_samples_only_when_intervals_overlap():
    cis = {"amem": (20.0, 60.0), "tsom": (10.0, 30.0)}
    assert f"n below {SMALL_N}" in comparison("Stale", cis, {"amem": 6, "tsom": 6})[0]
    assert "too small" not in comparison("Stale", cis, {"amem": 50, "tsom": 50})[0]


def test_comparison_needs_the_baseline():
    assert comparison("Stale", {"tsom": (10.0, 30.0)}, {"tsom": 50}) == []


def scenario_run(folder, stale):
    folder.mkdir()
    rows = [{"correct": not s, "stale": s, "hedged": False} for s in stale]
    (folder / "scenarios_summary.json").write_text(json.dumps({"per_scenario": rows}), encoding="utf-8")


def factcons_run(folder, stale, contradictions=None):
    folder.mkdir()
    rows = [{"correct": not s, "stale": s, "older": ["old"]} for s in stale]
    rows.append({"correct": True, "stale": False, "older": []})
    (folder / "sample_sh_6k.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    summary = {"contradictions": contradictions}
    (folder / "summary_factcons_sh_6k.json").write_text(json.dumps(summary), encoding="utf-8")


def test_scenario_table(tmp_path):
    scenario_run(tmp_path / "a", [True] * 40)
    scenario_run(tmp_path / "t", [False] * 40)
    text = make_tables({"amem": [tmp_path / "a"], "tsom": [tmp_path / "t"]})
    assert "| amem | 0.0% [0.0, 0.0] (n=40) | 100.0% [100.0, 100.0] (n=40) |" in text
    assert "Stale, tsom vs amem: lower" in text


def test_factconsolidation_stale_counts_only_changed_questions(tmp_path):
    factcons_run(tmp_path / "t", [True, False], {"final_rate": 1.5, "final_rate_without_consolidation": 11.0})
    text = make_tables({"tsom": [tmp_path / "t"]})
    assert "50.0%" in text and "(n=2)" in text and "(n=3)" in text
    assert "1.5 vs 11.0" in text


def test_systems_without_facts_show_no_contradiction_rate(tmp_path):
    factcons_run(tmp_path / "a", [True])
    assert "| n/a |" in make_tables({"amem": [tmp_path / "a"]})


def test_locomo_table_compares_with_the_baseline(tmp_path):
    for name, score in (("a", 40.0), ("t", 38.5)):
        (tmp_path / name).mkdir()
        scores = {"Overall": {"f1": score}, "Temporal": {"f1": score}}
        (tmp_path / name / "summary.json").write_text(json.dumps({"scores": scores}), encoding="utf-8")
    text = make_tables({"amem": [tmp_path / "a"], "tsom": [tmp_path / "t"]})
    assert "| tsom | 38.5 (-1.5) | 38.5 (-1.5) |" in text


def test_nothing_found_is_said_plainly(tmp_path):
    text = make_tables({"amem": [tmp_path / "none"]})
    for section in ("scenario", "FactConsolidation", "LoCoMo"):
        assert f"No {section} results found." in text


def test_run_argument():
    assert [p.name for p in parse_run("tsom=a,b")[1]] == ["a", "b"]
    with pytest.raises(argparse.ArgumentTypeError):
        parse_run("tsom")
