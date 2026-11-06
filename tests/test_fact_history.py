"""The get_fact_history tool, and the agent's choice of tool at question time."""

import pytest
from pydantic import ValidationError

from agent.assistant import Assistant
from harness import run_locomo, run_scenarios
from tsom.contracts import Fact, RunConfig
from tsom.llm import LLM
from tsom.memory import MemorySystem
from tsom.rag import RagMemory
from tsom.tools import TOOL_SPECS, MemoryTools


class Router(LLM):
    """Answers the tool choice with `reply`, and the keyword prompt without any past-tense cue."""

    def __init__(self, reply="TOOL: search_memory"):
        super().__init__("fake", "fake")
        self.reply, self.prompts = reply, []

    def _call(self, prompt, temperature):
        self.prompts.append(prompt)
        if prompt.startswith("Choose the memory tool"):
            return self.reply, 1, 1
        if prompt.startswith("Given the following question"):
            return "Alice, is", 1, 1
        return super()._call(prompt, temperature)


def config(system="tsom", **fields):
    return RunConfig(run_name="t", system=system, backend="fake", model="fake", **fields)


def moved(llm=None, **fields):
    llm = llm or LLM("fake", "fake")
    memory = MemorySystem(config(**fields), llm, sample_id="s0")
    memory.write("Alice is Lahore.", "t1")
    memory.write("Alice is Karachi.", "t2")
    return memory


def agent(llm, use_fact_history=True):
    return Assistant(MemoryTools(moved(llm)), llm, use_fact_history=use_fact_history)


def test_the_tool_is_listed_and_returns_the_whole_history():
    assert "get_fact_history" in [spec["function"]["name"] for spec in TOOL_SPECS]
    found = MemoryTools(moved()).call("get_fact_history", entity="alice")
    assert found["facts"].splitlines() == ["[PAST] Alice | is | Lahore (step 1, replaced at step 2)",
                                           "[CURRENT] Alice | is | Karachi (step 2)"]


def test_retracted_values_stay_out_of_the_history():
    memory = moved()
    memory.write("Alice is actually Dubai.", "t3")
    old = memory.store.fact_history("s0", "Alice", "is")[1]
    memory.store.update_fact_status(old.fact_id, "retracted")
    assert "Karachi" not in MemoryTools(memory).call("get_fact_history", entity="Alice")["facts"]


def test_an_unknown_entity_has_no_history():
    assert MemoryTools(moved()).call("get_fact_history", entity="Zed") == {"facts": ""}


def test_the_history_keeps_the_newest_lines_under_the_cap():
    memory = MemorySystem(config(), LLM("fake", "fake"), sample_id="s0")
    memory.temporal_retriever.max_facts = 2
    for step in range(5):
        memory.store.add_fact(Fact(sample_id="s0", note_id="n", entity="Alice", attribute=f"a{step}",
                                   value="v", observed_at=step))
    lines = memory.fact_history("Alice")["facts"].splitlines()
    assert [line.split("(step ")[1][0] for line in lines] == ["3", "4"]


@pytest.mark.parametrize("memory_cls,system", [(MemorySystem, "amem"), (RagMemory, "rag")])
def test_systems_without_facts_return_nothing(memory_cls, system):
    memory = memory_cls(config(system), LLM("fake", "fake"), sample_id="s0")
    assert MemoryTools(memory).call("get_fact_history", entity="Alice") == {"facts": ""}


def test_the_agent_adds_the_history_when_it_chooses_the_tool():
    llm = Router("TOOL: get_fact_history\nENTITY: Alice")
    context = agent(llm).answer("Where was Alice before?", 1, "Lahore")["context"]
    assert "[PAST] Alice | is | Lahore" in context
    assert context.startswith("Fact history:\n")


def test_the_agent_keeps_to_search_when_it_chooses_search():
    context = agent(Router()).answer("Where is Alice?", 1, "Karachi")["context"]
    assert "[PAST]" not in context and "Fact history" not in context


@pytest.mark.parametrize("reply", ["no idea", "TOOL: get_fact_history", "TOOL: get_fact_history\nENTITY: Zed"])
def test_an_unusable_choice_falls_back_to_search(reply):
    context = agent(Router(reply)).answer("Where was Alice?", 1, "Lahore")["context"]
    assert "Fact history" not in context


def test_the_choice_is_one_more_counted_llm_call():
    def calls(use_fact_history):
        llm = Router()
        assistant = agent(llm, use_fact_history)
        before = llm.usage.calls
        assistant.answer("Where is Alice?", 1, "Karachi")
        return llm.usage.calls - before

    assert calls(True) == calls(False) + 1


def test_with_the_flag_off_the_agent_is_never_asked():
    llm = Router()
    agent(llm, use_fact_history=False).answer("Where was Alice before?", 1, "Lahore")
    assert not any(p.startswith("Choose the memory tool") for p in llm.prompts)


def test_the_flag_needs_the_fact_layer():
    with pytest.raises(ValidationError):
        config("amem", use_fact_history=True)


def test_scenario_runs_use_the_flag(tmp_path):
    llm = Router()
    sc = run_scenarios.load_scenarios()[0]
    run_scenarios.run_scenario(config(use_fact_history=True), llm, sc)
    assert any(p.startswith("Choose the memory tool") for p in llm.prompts)


def test_locomo_runs_use_the_flag(tmp_path, monkeypatch):
    built = []
    real = run_locomo.Assistant

    def capture(*args, **kwargs):
        built.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(run_locomo, "Assistant", capture)
    run_locomo.run(RunConfig(run_name="s", system="tsom", backend="fake", model="fake", max_samples=1,
                             max_sessions=1, max_questions=1, output_dir=str(tmp_path),
                             use_fact_history=True))
    assert built[0]["use_fact_history"] is True
