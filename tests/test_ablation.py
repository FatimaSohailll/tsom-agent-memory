"""Ablation configs, tables, verdicts and the experiment runner."""

import json
import sys
from pathlib import Path

from harness.make_tables import bootstrap_ci, gap1_verdict, gap3_verdict, interaction_line, make_tables
from harness.scenarios import kn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import error_dump  # noqa: E402
import run_experiments  # noqa: E402


def test_exp_configs_differ_only_in_system_and_flags():
    configs = {p.stem: json.loads(p.read_text()) for p in (ROOT / "configs" / "exp").glob("*.json")}
    assert set(configs) == {"amem", "tsom_conflict", "tsom_time", "tsom_both"}
    flags = {"run_name", "system", "use_classifier", "use_validity"}
    shared = [{k: v for k, v in c.items() if k not in flags} for c in configs.values()]
    assert all(s == shared[0] for s in shared)
    assert shared[0] == {"backend": "ollama", "model": "qwen2.5:3b", "retrieve_k": 10, "use_gate": False}
    assert [(configs[n]["system"], configs[n]["use_classifier"], configs[n]["use_validity"])
            for n in ("amem", "tsom_conflict", "tsom_time", "tsom_both")] == [
        ("amem", False, False), ("tsom", True, False), ("tsom", False, True), ("tsom", True, True)]


# ---- verdicts on hand-made numbers ----

def actions_block(fr, mc):
    return {"false_retirements": kn(*fr), "missed_corrections": kn(*mc)}


def test_gap1_verdict():
    assert "Gap 1: supported" in gap1_verdict(actions_block((2, 40), (3, 30)), actions_block((9, 40), (3, 30)))
    worse_mc = gap1_verdict(actions_block((2, 40), (5, 30)), actions_block((9, 40), (3, 30)))
    assert "not supported" in worse_mc and "2/40" in worse_mc and "9/40" in worse_mc  # both counts printed
    assert "not supported" in gap1_verdict(actions_block((9, 40), (1, 30)), actions_block((9, 40), (3, 30)))
    assert "n is small" in gap1_verdict(actions_block((0, 5), (0, 5)), actions_block((2, 5), (1, 5)))
    assert "not available" in gap1_verdict(None, actions_block((1, 2), (1, 2)))
    assert "inconclusive" in gap1_verdict(actions_block((0, 0), (0, 0)), actions_block((0, 0), (0, 0)))


def test_gap3_verdict():
    amem, amem_ci = kn(15, 30), (33.3, 66.7)
    low, low_ci = kn(2, 30), (0.0, 16.7)
    assert "supported:" in gap3_verdict("tsom_time", low, low_ci, amem, amem_ci, 40.0, 41.0)
    assert ": supported" in gap3_verdict("tsom_time", low, low_ci, amem, amem_ci, 39.0, 41.0)  # exactly -2 is fine
    harm = gap3_verdict("tsom_time", low, low_ci, amem, amem_ci, 38.0, 41.0)
    assert "not supported" in harm and "drop 3.0" in harm
    overlap = gap3_verdict("tsom_time", kn(4, 10), (10.0, 70.0), kn(6, 10), (30.0, 90.0), 40.0, 40.0)
    assert "inconclusive (CIs overlap)" in overlap and "too small" in overlap
    assert "not supported" in gap3_verdict("tsom_both", kn(15, 30), amem_ci, amem, amem_ci, 40.0, 40.0)
    assert "no LoCoMo" in gap3_verdict("tsom_time", low, low_ci, amem, amem_ci, None, 41.0)
    assert "not available" in gap3_verdict("tsom_time", None, None, amem, amem_ci, None, None)


def test_interaction_line():
    assert "helps" in interaction_line(kn(1, 30), (0.0, 10.0), kn(12, 30), (20.0, 60.0))
    assert "inconclusive" in interaction_line(kn(3, 30), (0.0, 25.0), kn(5, 30), (5.0, 30.0))
    assert "does not lower" in interaction_line(kn(5, 30), (5.0, 30.0), kn(5, 30), (5.0, 30.0))


def test_bootstrap_ci_is_seeded_and_bounded():
    vals = [True] * 6 + [False] * 14
    lo, hi = bootstrap_ci(vals)
    assert 0 <= lo < 30 < hi <= 100 and bootstrap_ci(vals) == (lo, hi)
    assert bootstrap_ci([False] * 10) == (0.0, 0.0)
    assert bootstrap_ci([]) is None


# ---- make_tables on a tiny fixture ----

def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def actions_summary(fr, mc):
    block = {"scenarios": 4, "extraction_recall": kn(7, 8), "label_accuracy": kn(5, 7),
             "false_retirements": kn(*fr), "missed_corrections": kn(*mc), "parse_failures": 1}
    return {"overall": block, "per_kind": {"change": block}}


def scenarios_summary(stale, n=30):
    per = [{"id": f"s{i}", "kind": "change", "correct": i >= stale, "stale": i < stale, "hedged": False,
            "consistent": True, "retention": None} for i in range(n)]
    overall = {"scenarios": n, "stale": kn(stale, n), "correct": kn(n - stale, n), "hedged": kn(0, n),
               "consistent": kn(n, n), "retention": kn(0, 0)}
    return {"overall": overall, "per_kind": {"change": overall}, "per_scenario": per,
            "cost": {"writes": 10, "llm_calls_per_write": 3.0, "seconds_per_write": 4.0}}


def locomo_summary(temporal_f1, calls):
    cat = lambda f1: {"n": 10, "f1": f1, "bleu1": f1 - 5}  # noqa: E731
    return {"conversations": 1, "scores": {"Temporal": cat(temporal_f1), "Single-hop": cat(50.0),
                                           "Overall": {"n": 20, "f1": 45.0, "bleu1": 40.0}},
            "cost": {"llm_calls_per_write": calls, "seconds_per_write": 4.1}}


def test_make_tables_fixture(tmp_path):
    write(tmp_path / "actions/gap1_llm/actions_summary.json", actions_summary((1, 20), (2, 10)))
    write(tmp_path / "actions/gap1_lww/actions_summary.json", actions_summary((6, 20), (2, 10)))
    for row, stale in [("amem", 15), ("tsom_conflict", 12), ("tsom_time", 1), ("tsom_both", 1)]:
        write(tmp_path / f"scenarios/{row}/scenarios_summary.json", scenarios_summary(stale))
    write(tmp_path / "locomo/amem/summary.json", locomo_summary(40.0, 3.0))
    write(tmp_path / "locomo/tsom_time/summary.json", locomo_summary(39.5, 4.0))
    write(tmp_path / "locomo/tsom_both/summary.json", locomo_summary(35.0, 4.0))

    text = make_tables(tmp_path)
    assert text == (tmp_path / "tables.md").read_text()
    assert "| overall | gap1_llm | 7/8 (87.5%) | 5/7 (71.4%) | 1/20 (5.0%) | 2/10 (20.0%) | 1 |" in text
    assert "Gap 1: supported" in text
    assert "| amem | 15/30 (50.0%) [" in text and "| tsom_time | 1/30 (3.3%) [" in text
    assert "| rag |" not in text  # rows without results are left out
    assert "Gap 3 (tsom_time vs amem): supported" in text
    assert "Gap 3 (tsom_both vs amem): not supported" in text and "drop 5.0" in text
    assert "Interaction (tsom_both vs tsom_time): LLM labelling does not lower" in text
    assert "1 conversation(s), 20 questions per system" in text
    assert "| tsom_time | – | 39.5 / 34.5 | – | 50.0 / 45.0 | – | 45.0 / 40.0 | 4.00 | 4.10 |" in text


def test_make_tables_empty(tmp_path):
    text = make_tables(tmp_path)
    assert "No actions results found" in text and "No LoCoMo results found" in text


# ---- end to end with the fake backend ----

def test_run_experiments_scenarios_fake(tmp_path, capsys):
    run_experiments.main(["--suite", "scenarios", "--configs", "amem,tsom_both", "--limit", "2",
                          "--out", str(tmp_path), "--set", "backend=fake", "model=fake"])
    text = (tmp_path / "tables.md").read_text()
    assert "| amem | 0/2" in text and "| tsom_both | 0/2" in text
    assert (tmp_path / "scenarios" / "tsom_both" / "scenarios_summary.json").exists()

    capsys.readouterr()
    error_dump.main(["--run", str(tmp_path / "scenarios" / "tsom_both"), "--n", "1"])
    assert "verdict=wrong" in capsys.readouterr().out
