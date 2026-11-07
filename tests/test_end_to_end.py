"""End-to-end run on the first LoCoMo conversation with the fake LLM.

Checks the plumbing: replay, A-Mem writes, tool calls, answering, metrics,
traces and the summary table. Scores are meaningless with a fake model.
"""

import json

from harness.run_locomo import run
from tsom.contracts import RunConfig
from tsom.llm import LLM
from tsom.memory import MemorySystem
from tsom.tools import TOOL_SPECS, MemoryTools


def test_locomo_smoke(tmp_path):
    config = RunConfig(run_name="smoke", backend="fake", model="fake", max_samples=1,
                       max_sessions=2, max_questions=10, output_dir=str(tmp_path))
    summary = run(config)

    out = tmp_path / "smoke"
    rows = [json.loads(line) for line in (out / "sample_0.jsonl").read_text().splitlines()]
    ops = [json.loads(line) for line in (out / "ops_sample_0.jsonl").read_text().splitlines()]
    assert len(rows) == 10
    assert summary["scores"]["Overall"]["n"] == 10
    writes = [op for op in ops if op["op_type"] == "write"]
    assert writes and all(op["llm_calls"] >= 1 for op in writes)
    assert summary["cost"]["writes"] == len(writes)
    assert (out / "summary.md").exists()

    # Re-running skips the finished conversation and gives the same summary.
    again = run(config)
    assert again["scores"] == summary["scores"]


def test_tools_match_specs():
    memory = MemorySystem(RunConfig(run_name="t", backend="fake", model="fake"), LLM("fake", "fake"))
    tools = MemoryTools(memory)
    for spec in TOOL_SPECS:
        assert hasattr(tools, spec["function"]["name"])
    note_id = tools.call("write_memory", turn="Speaker Ali says : I live in Karachi", time="1 May 2023")["note_id"]
    assert memory.store.get_note(note_id) is not None
    assert "Karachi" in tools.call("search_memory", query="Karachi", k=1)["notes"]


def test_locomo_runs_only_the_chosen_conversations(tmp_path):
    from harness.run_locomo import load_config
    config = load_config("configs/smoke_fake.json", ["conversations=[2,0]", "max_sessions=1", "max_questions=2",
                                                     f"output_dir={tmp_path}", "run_name=pick"])
    assert config.conversations == [2, 0]
    summary = run(config)
    assert summary["conversation_ids"] == [2, 0] and summary["conversations"] == 2
    assert {p.name for p in (tmp_path / "pick").glob("sample_*.jsonl")} == {"sample_0.jsonl", "sample_2.jsonl"}
    rows = [json.loads(line) for line in (tmp_path / "pick" / "sample_2.jsonl").read_text().splitlines()]
    assert {r["sample"] for r in rows} == {2}


def test_an_unknown_conversation_is_refused(tmp_path):
    import pytest
    config = RunConfig(run_name="bad", backend="fake", model="fake", conversations=[10], output_dir=str(tmp_path))
    with pytest.raises(ValueError, match="no LoCoMo conversation"):
        run(config)

