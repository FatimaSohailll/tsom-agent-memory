"""Prompts for the temporal layer. Plain-text output only (no JSON mode)."""

FACT_EXTRACTION_PROMPT = """Extract the atomic facts stated in the text. Each fact is (entity, attribute, value).
Rules:
- One fact per block. Split compound sentences.
- Replace pronouns with the named entity when clear.
- ATTRIBUTE: short snake_case relation, e.g. capital_of, citizen_of, works_at, spouse, head_of_state, located_in.
- VALID_FROM / VALID_UNTIL: only if the text explicitly states a date or period, otherwise write NONE. Never guess.
- If there are no facts, output exactly: NO_FACTS

Respond using EXACTLY this format, one block per fact:
FACT 1:
ENTITY: ...
ATTRIBUTE: ...
VALUE: ...
VALID_FROM: ...
VALID_UNTIL: ...

Text:
{text}"""

# Batch mode: several numbered statements per call; each fact names its source.
FACT_BATCH_EXTRACTION_PROMPT = """Extract the atomic facts stated in each numbered statement. Each fact is (entity, attribute, value).
Rules:
- One fact per block. Split compound sentences.
- Replace pronouns with the named entity when clear.
- ATTRIBUTE: short snake_case relation, e.g. capital_of, citizen_of, works_at, spouse, head_of_state, located_in.
- VALID_FROM / VALID_UNTIL: only if the statement explicitly states a date or period, otherwise write NONE. Never guess.
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

Statements:
{statements}"""
