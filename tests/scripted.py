"""A scripted LLM for offline tests."""

from tsom.facts import PROMPT_HEAD
from tsom.llm import LLM, fake_completion


class ScriptedLLM(LLM):
    """Answers extraction prompts from a script; everything else goes to fake_completion.

    Only _call is replaced, so calls and tokens are still counted in usage.
    """

    def __init__(self, script=()):
        super().__init__("fake", "scripted")
        self.script = list(script)
        self.prompts, self.temperatures = [], []

    def _call(self, prompt, temperature):
        if prompt.startswith(PROMPT_HEAD):
            self.prompts.append(prompt)
            self.temperatures.append(temperature)
            return self.script.pop(0), len(prompt.split()), 5
        return fake_completion(prompt), len(prompt.split()), 5
