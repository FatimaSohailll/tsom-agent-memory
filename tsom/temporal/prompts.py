"""Prompts for the temporal layer. Plain-text output only (no JSON mode)."""

FACT_EXTRACTION_PROMPT = """Extract the atomic facts stated in the text. Each fact is (entity, attribute, value).
Rules:
- One fact per block. Split compound sentences.
- Replace pronouns with the named entity when clear.
- ATTRIBUTE: short snake_case relation, e.g. capital_of, citizen_of, works_at, spouse, head_of_state, located_in.
- VALID_FROM / VALID_UNTIL: only if the text explicitly states a date or period, otherwise write NONE. Never guess.
- TENSE: PAST if the wording says it is over or earlier ("worked", "used to", "formerly", "was ... until"); PRESENT if it describes the current state ("now", "is", "currently", present tense); UNKNOWN otherwise. TENSE describes the wording only. It never changes VALID_FROM or VALID_UNTIL, which stay NONE unless a date is written.
- If there are no facts, output exactly: NO_FACTS

Respond using EXACTLY this format, one block per fact:
FACT 1:
ENTITY: ...
ATTRIBUTE: ...
VALUE: ...
VALID_FROM: ...
VALID_UNTIL: ...
TENSE: ...

Example text: In 2018 Ali worked at Careem.
FACT 1:
ENTITY: Ali
ATTRIBUTE: works_at
VALUE: Careem
VALID_FROM: 2018
VALID_UNTIL: NONE
TENSE: PAST

Example text: Ali now works at Microsoft.
FACT 1:
ENTITY: Ali
ATTRIBUTE: works_at
VALUE: Microsoft
VALID_FROM: NONE
VALID_UNTIL: NONE
TENSE: PRESENT

{attribute_hint}Text:
{text}"""

# Batch mode: several numbered statements per call; each fact names its source.
FACT_BATCH_EXTRACTION_PROMPT = """Extract the atomic facts stated in each numbered statement. Each fact is (entity, attribute, value).
Rules:
- One fact per block. Split compound sentences.
- Replace pronouns with the named entity when clear.
- ATTRIBUTE: short snake_case relation, e.g. capital_of, citizen_of, works_at, spouse, head_of_state, located_in.
- VALID_FROM / VALID_UNTIL: only if the statement explicitly states a date or period, otherwise write NONE. Never guess.
- TENSE: PAST if the wording says it is over or earlier ("worked", "used to", "formerly", "was ... until"); PRESENT if it describes the current state ("now", "is", "currently", present tense); UNKNOWN otherwise. TENSE describes the wording only. It never changes VALID_FROM or VALID_UNTIL, which stay NONE unless a date is written.
- SOURCE: the number of the statement the fact comes from.
- If there are no facts at all, output exactly: NO_FACTS

Respond using EXACTLY this format, one block per fact:
FACT 1:
SOURCE: ...
ENTITY: ...
ATTRIBUTE: ...
VALUE: ...
VALID_FROM: ...
VALID_UNTIL: ...
TENSE: ...

Example statements:
1. In 2018 Ali worked at Careem.
2. Ali now works at Microsoft.
FACT 1:
SOURCE: 1
ENTITY: Ali
ATTRIBUTE: works_at
VALUE: Careem
VALID_FROM: 2018
VALID_UNTIL: NONE
TENSE: PAST
FACT 2:
SOURCE: 2
ENTITY: Ali
ATTRIBUTE: works_at
VALUE: Microsoft
VALID_FROM: NONE
VALID_UNTIL: NONE
TENSE: PRESENT

{attribute_hint}Statements:
{statements}"""

UPDATE_CLASSIFY_PROMPT = """A memory already contains this fact:
  {old_lines}
A new fact arrives:
  {entity} | {attribute} | {new_value} (learned at step {new_observed}, session {new_session}, valid from: {new_from}, valid until: {new_until}, tense: {new_tense})
{extra_context}

Classify how the new fact relates to the old one:
- REPLACE: the value changed over time; the old value was true before.
- CORRECT: the old value was wrong or never true.
- ADD: both can be true at the same time. Relations such as languages spoken, children, memberships and hobbies can hold several values at once.
A past-tense fact about an earlier period does not replace a current fact.
If the relation normally has one value at a time, choose REPLACE; if a person or thing can have several, choose ADD.

Respond using EXACTLY this format:
OPERATION: <REPLACE | CORRECT | ADD>
REASON: <one short sentence>"""


def attribute_hint(names) -> str:
    """The prompt line offering attribute names already in use ("" if there are none)."""
    if not names:
        return ""
    return ("Attribute names already in use (reuse one of them if it means the same relation, "
            "otherwise use a new short snake_case name): " + ", ".join(names) + "\n\n")


def build_extraction_prompt(text: str, known_attributes=None) -> str:
    return FACT_EXTRACTION_PROMPT.format(text=text, attribute_hint=attribute_hint(known_attributes))


def build_batch_extraction_prompt(statements: str, known_attributes=None) -> str:
    return FACT_BATCH_EXTRACTION_PROMPT.format(statements=statements,
                                               attribute_hint=attribute_hint(known_attributes))
