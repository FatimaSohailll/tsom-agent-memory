"""The conflict gate (Gap 2), its cost metrics, the tau sweep and the gate table."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from harness.run_locomo import cost_summary, load_config, run, select_samples, to_markdown
from tsom.amem_bridge import RobustAgenticMemorySystem
from tsom.contracts import RunConfig
from tsom.gated_amem import GatedAgenticMemory, decide
from tsom.llm import fake_completion
from tsom.memory import MemorySystem

from scripted import ScriptedLLM

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import make_gate_table  # noqa: E402
import sweep_tau  # noqa: E402

EVOLUTION_HEAD = "You are an AI memory evolution agent"
DATE = "1:56 pm on 8 May, 2023"
QUIET = ["Haha see you tomorrow then", "The weather was lovely this afternoon",
         "We watched a long movie yesterday", "Dinner tonight was really tasty"]


class GateLLM(ScriptedLLM):
    """Extraction from the script; A-Mem's evolution decision is always STRENGTHEN."""

    def __init__(self, script=()):
        super().__init__(script)
        self.all_prompts = []

    def _call(self, prompt, temperature):
        self.all_prompts.append(prompt)
        if prompt.startswith(EVOLUTION_HEAD):
            return "DECISION: STRENGTHEN\nREASON: test", len(prompt.split()), 5
        return super()._call(prompt, temperature)

    def evolution_prompts(self):
        return [p for p in self.all_prompts if p.startswith(EVOLUTION_HEAD)]


def gated(*script, llm_cls=GateLLM, **flags):
    flags = {"use_classifier": True, "use_gate": True, "tau": 0.99, **flags}
    config = RunConfig(run_name="t", system="tsom", backend="fake", model="fake", **flags)
    llm = llm_cls(script)
    return MemorySystem(config, llm), llm


def last_write(memory):
    return [op for op in memory.store.ops() if op.op_type == "write"][-1]


# ---- the rule ----

def test_decide_rule():
    assert decide([], None, 0.8).reason == "no_neighbours"
    assert not decide(["corrective"], None, 0.8).run_evolution  # nothing to evolve against
    assert decide(["additive", "corrective"], 0.1, 0.8).reason == "conflict"
    assert decide(["contradictory"], 0.1, 0.8).run_evolution
    d = decide(["additive"], 0.8, 0.8)  # similarity == tau evolves
    assert d.run_evolution and d.reason == "neighbour" and d.max_similarity == 0.8
    d = decide(["additive", "redundant"], 0.79, 0.8)
    assert not d.run_evolution and d.reason == "skipped"


# ---- the gate inside MemorySystem ----

def test_quiet_turns_skip_evolution_and_link_by_embedding():
    memory, llm = gated("NONE", "NONE", "NONE", "NONE")
    for text in QUIET[:3]:
        memory.write(text, DATE, speaker="Ali")
    assert last_write(memory).gate == "skipped"
    sims = memory.amem.similarities(QUIET[3])
    memory.write(QUIET[3], DATE, speaker="Ali")
    op = last_write(memory)
    assert op.gate == "skipped" and op.llm_calls == 2  # extraction + A-Mem analysis only
    assert op.neighbour_similarity == pytest.approx(float(sims.max()))
    assert not op.evolved and not llm.evolution_prompts()
    note = list(memory.amem.memories.values())[-1]
    assert note.links == [int(i) for i in np.argsort(sims)[::-1][:3]]
    assert sorted(note.links) == [0, 1, 2]
    first = [op for op in memory.store.ops() if op.op_type == "write"][0]
    assert first.gate == "no_neighbours" and first.neighbour_similarity is None


def test_link_k_zero_gives_no_links():
    memory, _ = gated("NONE", "NONE", gate_link_k=0)
    memory.write(QUIET[0], DATE, speaker="Ali")
    memory.write(QUIET[1], DATE, speaker="Ali")
    assert last_write(memory).gate == "skipped"
    assert list(memory.amem.memories.values())[-1].links == []


@pytest.mark.parametrize("second", ["FACT: lives_in | Lahore | corrective | Karachi",
                                    "FACT: lives_in | Lahore | contradictory | none"])
def test_conflict_labels_run_evolution(second):
    memory, llm = gated("FACT: lives_in | Karachi | additive | none", second)
    memory.write("I live in Karachi", DATE, speaker="Ali")
    memory.write("Lahore is where I am", DATE, speaker="Ali")
    op = last_write(memory)
    assert op.gate == "conflict" and op.labels[0] in ("corrective", "contradictory")
    assert len(llm.evolution_prompts()) == 1 and op.llm_calls >= 3


def test_neighbour_at_tau_runs_evolution():
    memory, llm = gated("NONE", "NONE")
    memory.write(QUIET[0], DATE, speaker="Ali")
    memory.amem.tau = float(memory.amem.similarities(QUIET[1]).max())  # exactly at tau
    memory.write(QUIET[1], DATE, speaker="Ali")
    assert last_write(memory).gate == "neighbour" and len(llm.evolution_prompts()) == 1


def test_always_open_gate_matches_ungated():
    script = ["NONE", "FACT: lives_in | Karachi | additive | none", "NONE",
              "FACT: lives_in | Lahore | corrective | Karachi", "NONE"]
    texts = QUIET[:2] + ["I live in Karachi", "I moved to Lahore", QUIET[3]]
    results = []
    for use_gate in (False, True):
        memory, _ = gated(*script, llm_cls=ScriptedLLM, use_gate=use_gate, tau=0.0)
        for text in texts:
            memory.write(text, DATE, speaker="Ali")
        writes = [op for op in memory.store.ops() if op.op_type == "write"]
        notes = [(n.content, n.links, n.tags, n.context, n.keywords) for n in memory.amem.memories.values()]
        results.append(([op.llm_calls for op in writes], [op.evolved for op in writes], notes))
    assert results[0] == results[1]
    assert any(results[0][1]) and max(results[0][0]) > 2  # evolution really ran somewhere


def test_ungated_uses_plain_amem_and_logs_no_gate():
    memory, _ = gated("NONE", use_gate=False)
    assert type(memory.amem) is RobustAgenticMemorySystem
    memory.write(QUIET[0], DATE, speaker="Ali")
    op = last_write(memory)
    assert op.gate is None and op.neighbour_similarity is None
    assert isinstance(gated(use_gate=True)[0].amem, GatedAgenticMemory)


@pytest.mark.parametrize("system", ["amem", "rag"])
def test_gate_needs_tsom(system):
    with pytest.raises(ValueError):
        RunConfig(run_name="t", system=system, use_gate=True)


def test_session_is_logged():
    memory, _ = gated("NONE")
    memory.current_session = 3
    memory.write(QUIET[0], DATE, speaker="Ali")
    assert last_write(memory).session == 3


# ---- harness metrics ----

def op(sample, session, calls, seconds, gate=None, sim=None):
    return {"op_type": "write", "sample_id": str(sample), "session": session, "llm_calls": calls,
            "latency_s": seconds, "evolved": gate in ("conflict", "neighbour"), "gate": gate,
            "neighbour_similarity": sim}


def test_cost_summary_percentiles_sessions_and_gate():
    ops = [op(0, 1, 2, 1.0, "no_neighbours"), op(0, 1, 5, 4.0, "conflict", 0.2),
           op(0, 2, 2, 1.0, "skipped", 0.4), op(1, 1, 4, 3.0, "neighbour", 0.9)]
    cost = cost_summary([{"prompt_tokens": 10}], ops)
    latencies = [1.0, 4.0, 1.0, 3.0]
    assert cost["write_latency_p50"] == pytest.approx(np.percentile(latencies, 50))
    assert cost["write_latency_p95"] == pytest.approx(np.percentile(latencies, 95))
    assert [(r["conversation"], r["session"], r["llm_calls"], r["skipped"]) for r in cost["per_session"]] == \
        [(0, 1, 7, 0), (0, 2, 2, 1), (1, 1, 4, 0)]
    assert cost["llm_calls_per_session"] == pytest.approx(13 / 3)
    gate = cost["gate"]
    assert gate["decisions"] == {"no_neighbours": 1, "conflict": 1, "neighbour": 1, "skipped": 1}
    assert gate["skip_rate"] == 0.25
    assert gate["similarity"]["p50"] == pytest.approx(np.percentile([0.2, 0.4, 0.9], 50))
    text = to_markdown({"run_name": "g", "system": "tsom", "model": "m", "conversations": 2,
                        "conversation_ids": [0, 1], "seed": 42, "scores": {}, "cost": cost})
    assert "Conversations: 0, 1 (seed 42)" in text and "Gate skip rate | 25.0%" in text


def test_cost_summary_without_sessions_or_gate():
    old = [{"op_type": "write", "sample_id": "0", "llm_calls": 3, "latency_s": 2.0, "evolved": True}]
    cost = cost_summary([], old)
    assert cost["llm_calls_per_session"] is None and cost["per_session"] == [] and "gate" not in cost
    assert {"writes", "llm_calls_per_write", "seconds_per_write", "evolution_rate",
            "answer_prompt_tokens_per_question"} <= set(cost)


def test_select_samples_and_config_overrides():
    config = load_config(ROOT / "configs" / "gate" / "tsom_full_gated.json", ["seed=7", "tau=0.7"], [2, 0])
    assert config.seed == 7 and config.tau == 0.7 and config.conversation_ids == [2, 0] and config.use_gate
    assert select_samples(["a", "b", "c"], config) == [(2, "c"), (0, "a")]
    assert select_samples(["a", "b", "c"], config.model_copy(update={"conversation_ids": None,
                                                                     "max_samples": 2})) == [(0, "a"), (1, "b")]
    with pytest.raises(ValueError):
        select_samples(["a"], config)


def test_gated_locomo_smoke(tmp_path):
    config = load_config(ROOT / "configs" / "gate" / "tsom_full_gated.json",
                         ["backend=fake", "model=fake", "max_sessions=2", "max_questions=3",
                          f"output_dir={tmp_path}", "seed=5"], [1])
    summary = run(config)
    assert summary["conversation_ids"] == [1] and summary["seed"] == 5
    assert json.loads((tmp_path / "tsom_full_gated" / "config.json").read_text())["seed"] == 5
    cost = summary["cost"]
    assert sum(cost["gate"]["decisions"].values()) == cost["writes"]
    assert {r["session"] for r in cost["per_session"]} == {1, 2}
    ops = [json.loads(line) for line in (tmp_path / "tsom_full_gated" / "ops_sample_1.jsonl").read_text().splitlines()]
    assert all(o["session"] in (1, 2) for o in ops if o["op_type"] == "write")


# ---- tau sweep helpers ----

def summary(calls, f1, temporal=None, skip=None, p50=1.0, p95=2.0):
    scores = {"Overall": {"n": 10, "f1": f1, "bleu1": 0}}
    if temporal is not None:
        scores["Temporal"] = {"n": 3, "f1": temporal, "bleu1": 0}
    cost = {"llm_calls_per_write": calls, "write_latency_p50": p50, "write_latency_p95": p95}
    if skip is not None:
        cost["gate"] = {"skip_rate": skip}
    return {"scores": scores, "cost": cost}


def test_sweep_rows_and_suggestion():
    rows = sweep_tau.sweep_rows(summary(4.0, 30.0, 20.0),
                                {0.9: summary(3.8, 29.5, skip=0.1), 0.6: summary(2.5, 28.0, skip=0.6),
                                 0.7: summary(3.0, 27.9, skip=0.4)})
    assert [r["tau"] for r in rows] == [None, 0.6, 0.7, 0.9]
    assert rows[1]["calls_change"] == pytest.approx(-1.5) and rows[1]["f1_change"] == pytest.approx(-2.0)
    assert rows[1]["within_margin"] and not rows[2]["within_margin"]  # -2.0 is within, -2.1 is not
    assert sweep_tau.suggest(rows) == 0.6
    text = sweep_tau.to_markdown(rows, 0.6, sweep_tau.similarity_distribution([0.1, 0.5, 0.9]))
    assert "Suggested tau: 0.6" in text and "never re-tuned on a test conversation" in text
    assert "2-3 points are inside the noise" in text and "p50 0.500" in text


def test_sweep_without_a_run_in_margin():
    rows = sweep_tau.sweep_rows(summary(4.0, 30.0), {0.8: summary(2.0, 20.0, skip=0.5)})
    assert sweep_tau.suggest(rows) is None
    assert "cost-accuracy curve above is the result" in sweep_tau.to_markdown(rows, None, {})


def test_sweep_refuses_conversation_override():
    with pytest.raises(SystemExit):
        sweep_tau.main(["--set", "conversation_ids=[1]"])


def test_sweep_plot(tmp_path):
    pytest.importorskip("matplotlib")
    rows = sweep_tau.sweep_rows(summary(4.0, 30.0), {0.7: summary(3.0, 29.0, skip=0.3),
                                                     0.8: summary(3.5, 29.5, skip=0.2)})
    sweep_tau.plot(rows, tmp_path / "p.png")
    assert (tmp_path / "p.png").stat().st_size > 0


# ---- gate table ----

def metrics(calls, p50, p95, f1):
    return {"calls_per_write": calls, "latency_p50": p50, "latency_p95": p95, "f1_Overall": f1}


def test_aggregate_mean_and_sample_std():
    agg = make_gate_table.aggregate([metrics(1, 1, 1, 10), metrics(2, 1, 1, 20), metrics(3, 1, 1, 30)])
    assert agg["calls_per_write"] == {"mean": 2, "std": 1.0, "n": 3}
    one = make_gate_table.aggregate([metrics(1, 1, 1, 10)])
    assert one["calls_per_write"]["std"] is None
    text = make_gate_table.to_markdown(one, one, 1, 1, [], "x")
    assert "n=1, no std" in text


def verdict_for(ungated, gated, n=3):
    u, g = make_gate_table.aggregate([ungated] * n), make_gate_table.aggregate([gated] * n)
    checks = make_gate_table.h2_checks(u, g)
    return checks, make_gate_table.verdict(checks, n, n)


def test_h2_supported():
    checks, text = verdict_for(metrics(4, 2, 5, 30), metrics(3, 1, 4, 29))
    assert text == "H2 supported" and all(c["passed"] for c in checks)


def test_h2_partly_names_the_failed_check():
    checks, text = verdict_for(metrics(4, 2, 5, 30), metrics(3, 1, 6, 29))
    assert text == "H2 partly supported; failed: (b) lower p50 and p95 write latency"
    assert "p95 gated 6.00s vs ungated 5.00s (higher)" in checks[1]["detail"]


def test_h2_not_supported_and_few_runs():
    checks, text = verdict_for(metrics(4, 2, 5, 30), metrics(5, 3, 6, 25), n=2)
    assert text == "H2 not supported (fewer than 3 runs: indicative only)"
    assert "dropped by more than 2 points" in checks[2]["detail"]


def test_h2_ties_are_no_difference():
    checks, text = verdict_for(metrics(4, 2, 5, 30), metrics(4, 2, 5, 30))
    assert not checks[0]["passed"] and "(no difference)" in checks[0]["detail"]
    assert "worse" not in json.dumps(checks) and "higher" not in json.dumps(checks)
    assert checks[2]["passed"] and "(no difference)" in checks[2]["detail"]
    assert text.startswith("H2 partly supported; failed: (a) fewer LLM calls per write; (b)")


def test_make_gate_table_main(tmp_path):
    folders = []
    for name, s in (("u", summary(4.0, 30.0)), ("g", summary(3.0, 29.0, skip=0.4))):
        (tmp_path / name).mkdir()
        (tmp_path / name / "summary.json").write_text(json.dumps(s))
        folders.append(str(tmp_path / name))
    make_gate_table.main(["--ungated", folders[0], "--gated", folders[1], "--out", str(tmp_path)])
    data = json.loads((tmp_path / "gate_table.json").read_text(encoding="utf-8"))
    assert data["verdict"].endswith("(fewer than 3 runs: indicative only)")
    assert "n=1, no std" in (tmp_path / "gate_table.md").read_text(encoding="utf-8")


def test_fake_completion_still_drives_evolution():
    # ScriptedLLM falls back to fake_completion, whose evolution answers the gate tests rely on.
    assert fake_completion(EVOLUTION_HEAD + " x").startswith("DECISION:")
