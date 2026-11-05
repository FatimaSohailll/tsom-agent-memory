"""Temperature and seed control: off by default, and it reaches A-Mem's own calls."""

from types import SimpleNamespace

import pytest

from harness.factcons_answer import answer_question
from tsom.contracts import RunConfig
from tsom.llm import LLM, MAX_TOKENS
from tsom.memory import MemorySystem


class Ollama:
    def __init__(self):
        self.sent = None

    def chat(self, **kwargs):
        self.sent = kwargs
        return {"message": {"content": "ok"}}


class OpenAI:
    def __init__(self):
        self.sent = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.sent = kwargs
        message = SimpleNamespace(content="ok")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def connected(backend, client, **fields):
    llm = LLM("fake", "m", **fields)
    llm.backend, llm.client = backend, client
    return llm


class Recording(LLM):
    def __init__(self, **fields):
        super().__init__("fake", "fake", **fields)
        self.temperatures = []

    def _call(self, prompt, temperature):
        self.temperatures.append(temperature)
        return super()._call(prompt, temperature)


def config(**fields):
    return RunConfig(run_name="t", backend="fake", model="fake", **fields)


def test_requests_are_unchanged_by_default():
    ollama, openai = Ollama(), OpenAI()
    connected("ollama", ollama).get_completion("hi", temperature=0.3)
    assert ollama.sent["options"] == {"temperature": 0.3, "num_predict": MAX_TOKENS}
    connected("openai", openai).get_completion("hi", temperature=0.3)
    assert openai.sent["temperature"] == 0.3 and "seed" not in openai.sent


def test_a_forced_temperature_replaces_the_callers():
    ollama = Ollama()
    connected("ollama", ollama, temperature=0.0).get_completion("hi", temperature=0.7)
    assert ollama.sent["options"]["temperature"] == 0.0


@pytest.mark.parametrize("backend", ["openai", "vllm"])
def test_seed_reaches_openai_style_backends(backend):
    openai = OpenAI()
    connected(backend, openai, seed=7).get_completion("hi")
    assert openai.sent["seed"] == 7


def test_seed_reaches_ollama():
    ollama = Ollama()
    connected("ollama", ollama, seed=7).get_completion("hi")
    assert ollama.sent["options"]["seed"] == 7


def test_config_builds_the_llm():
    llm = LLM.from_config(config(temperature=0, seed=3, send_seed=True))
    assert (llm.temperature, llm.seed) == (0, 3)


def test_config_keeps_the_seed_local_unless_asked():
    llm = LLM.from_config(config(seed=3))
    assert llm.temperature is None and llm.seed is None


@pytest.mark.parametrize("system", ["amem", "tsom"])
def test_forced_temperature_covers_amem_and_the_fact_layer(system):
    llm = Recording(temperature=0.0)
    memory = MemorySystem(config(system=system), llm, sample_id="s0")
    memory.write("Alice is in Lahore.", "t1")
    memory.write("Alice is in Karachi.", "t2")
    assert llm.temperatures and set(llm.temperatures) == {0.0}


def test_amem_keeps_its_own_temperature_by_default():
    llm = Recording()
    MemorySystem(config(system="amem"), llm, sample_id="s0").write("Alice is in Lahore.", "t1")
    assert 0.7 in llm.temperatures


def test_forced_temperature_covers_answers():
    llm = Recording(temperature=0.0)
    memory = MemorySystem(config(system="amem"), llm, sample_id="s0")
    answer_question(memory, llm, "Where is Alice?", 5, temperature=0.7)
    assert llm.temperatures == [0.0]
