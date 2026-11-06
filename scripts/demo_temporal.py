"""Temporal-memory demo: eight statements about Ali, then four questions.

    python scripts/demo_temporal.py                    # configs/tsom_qwen2.5-3b.json (Ollama)
    python scripts/demo_temporal.py --backend fake     # offline, no model needed
    python scripts/demo_temporal.py --rules off        # multi-valued rules off: the LLM prompt alone decides
    python scripts/demo_temporal.py --classifier llm_only   # the LLM decides every differing value (no rules)
    python scripts/demo_temporal.py --raw              # also print raw extractor/classifier responses
    python scripts/demo_temporal.py --verbose          # also show library log messages

Every statement goes through MemorySystem.write with system="tsom". After each
write it prints the facts extracted, the operation chosen for each fact, the
facts it touched, the classifier's reason (when the LLM was asked), and any
note_versions rows A-Mem's evolution created. The probes below only wrap the
library's methods to observe them; no library code is changed.

With --backend fake, fact extraction is scripted for these eight statements
(the stock fake only understands "X is Y"); the classifier verdicts and the
answers are the fake backend's, so they are placeholders, not model output.
"""

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harness.factcons_answer import clean_prediction  # noqa: E402
from tsom.contracts import RunConfig  # noqa: E402
from tsom.llm import LLM  # noqa: E402
from tsom.memory import FACTS_HEADER, MemorySystem  # noqa: E402
from tsom.temporal.parsers import parse_update_classification  # noqa: E402
from tsom.temporal.retriever import classify_query  # noqa: E402

# The demo's own answer prompt. The FactConsolidation prompt (harness/factcons_answer.py) forces a
# single entity, which cannot answer "what languages" questions; it is left unchanged so benchmark
# runs stay comparable. The opening sentence is shared, so the fake backend answers it the same way.
LIST_ANSWER_PROMPT = """Answer the question using the memory below. Facts are numbered; when facts
conflict, the one with the larger number (or marked CURRENT) is the latest
and must be used.
If the question asks for everything of some kind and several CURRENT facts give different values
for it, list all of those values separated by commas. Otherwise answer with only the entity name.
No explanation.
Memory:
{context}
Question: {question}
Answer:"""

# (statement, session date, expected operation)
STEPS = [
    ("Ali works at Google.", "2020-01-10", "ADD"),
    ("Ali lives in Lahore.", "2020-02-01", "ADD"),
    ("Ali works at Google.", "2020-03-01", "REAFFIRM"),
    ("Ali now works at Microsoft.", "2024-03-15", "REPLACE"),
    ("Correction: Ali never lived in Lahore, he lives in Karachi.", "2024-04-01", "CORRECT"),
    ("Ali speaks Urdu.", "2024-05-01", "ADD"),
    ("Ali also speaks English.", "2024-05-02", "ADD"),
    ("In 2018 Ali worked at Careem.", "2024-06-01", "ADD_HISTORICAL"),
]

# (question, words the answer should contain, words it must not contain)
QUERIES = [
    ("Where does Ali work?", ["Microsoft"], []),
    ("Where did Ali work in 2020?", ["Google"], []),
    ("Where does Ali live?", ["Karachi"], ["Lahore"]),
    ("What languages does Ali speak?", ["Urdu", "English"], []),
]

# Stand-in extractor for --backend fake: statement -> (entity, attribute, value, valid_from, tense)
SCRIPTED_FACTS = {
    "Ali works at Google.": ("Ali", "works_at", "Google", "NONE", "PRESENT"),
    "Ali lives in Lahore.": ("Ali", "lives_in", "Lahore", "NONE", "PRESENT"),
    "Ali now works at Microsoft.": ("Ali", "works_at", "Microsoft", "NONE", "PRESENT"),
    "Correction: Ali never lived in Lahore, he lives in Karachi.": ("Ali", "lives_in", "Karachi", "NONE", "PRESENT"),
    "Ali speaks Urdu.": ("Ali", "speaks", "Urdu", "NONE", "PRESENT"),
    "Ali also speaks English.": ("Ali", "speaks_language", "English", "NONE", "PRESENT"),  # drifts, as the real model did
    "In 2018 Ali worked at Careem.": ("Ali", "works_at", "Careem", "2018", "PAST"),
}


class ScriptedFakeLLM(LLM):
    """The fake backend, plus scripted fact extraction for this demo's statements."""

    def __init__(self):
        super().__init__("fake", "fake")

    def get_completion(self, prompt, temperature=0.7):
        if prompt.startswith("Extract the atomic facts stated in the text."):
            self.usage.calls += 1
            fact = SCRIPTED_FACTS.get(prompt.split("\nText:\n")[-1].strip())
            if fact is None:
                return "NO_FACTS"
            e, a, v, start, tense = fact
            return (f"FACT 1:\nENTITY: {e}\nATTRIBUTE: {a}\nVALUE: {v}\n"
                    f"VALID_FROM: {start}\nVALID_UNTIL: NONE\nTENSE: {tense}")
        return super().get_completion(prompt, temperature)


class Probe:
    """Records what the temporal step did, by wrapping the library's methods."""

    def __init__(self, mem, llm):
        self.extracted, self.decisions, self.raw_extractions, self.hints = [], [], [], []
        extract, classify, apply = mem.extractor.extract, mem.classifier.classify, mem.consolidator.apply
        complete = llm.get_completion

        def probe_extract(text, *args, **kwargs):
            self.hints.append(list(args[0]) if args else kwargs.get("known_attributes") or [])
            facts = extract(text, *args, **kwargs)
            self.extracted += facts
            return facts

        def probe_classify(*args, **kwargs):
            op, old = classify(*args, **kwargs)
            self.decisions.append({"new": args[1], "op": op, "old": old, "applied": op,
                                   **{k: mem.classifier.last_decision[k] for k in ("source", "reason", "raw")}})
            return op, old

        def probe_apply(*args, **kwargs):
            fact = apply(*args, **kwargs)
            if self.decisions:
                self.decisions[-1]["applied"] = mem.consolidator.last_op
            return fact

        def probe_complete(prompt, temperature=0.7):
            response = complete(prompt, temperature)
            if prompt.startswith("Extract the atomic facts stated in"):
                self.raw_extractions.append(response)
            return response

        mem.extractor.extract = probe_extract
        mem.classifier.classify = probe_classify
        mem.consolidator.apply = probe_apply
        llm.get_completion = probe_complete

    def reset(self):
        self.extracted, self.decisions, self.raw_extractions, self.hints = [], [], [], []


def table(headers, rows):
    rows = [[str(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    line = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()
    return "\n".join([line(headers), line(["-" * w for w in widths])] + [line(r) for r in rows])


def short(fact_id):
    return fact_id[:8] if fact_id else "-"


def describe(f):
    return f"{f.entity} | {f.attribute} | {f.value} [{short(f.fact_id)}]"


def header(text):
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def load_config(args) -> RunConfig:
    path = Path(args.config)
    if not path.exists() and not path.is_absolute():
        path = REPO_ROOT / path
    data = json.loads(path.read_text())
    data["system"] = "tsom"
    data["run_name"] = "demo_temporal"
    # Set explicitly, whatever the config says: --rules on|off decides.
    data["seed_multivalued"] = data["additive_cue"] = args.rules == "on"
    data["classifier_mode"] = args.classifier
    if args.backend:
        data["backend"] = args.backend
        if args.backend == "fake" and not args.model:
            data["model"] = "fake"
    if args.model:
        data["model"] = args.model
    return RunConfig(**data)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/tsom_qwen2.5-3b.json")
    parser.add_argument("--backend", help="override the config's backend, e.g. fake")
    parser.add_argument("--model", help="override the config's model")
    parser.add_argument("--rules", choices=["on", "off"], default="on",
                        help="on: seed_multivalued and additive_cue enabled (default); off: both disabled")
    parser.add_argument("--classifier", choices=["rules_llm", "llm_only", "newest_wins"], default="rules_llm",
                        help="rules_llm: deterministic rules first, LLM for the rest (default); "
                             "llm_only: the LLM decides every differing value (--rules has no effect); "
                             "newest_wins: no classifier, every differing value replaces (--rules has no effect)")
    parser.add_argument("--raw", action="store_true",
                        help="print the raw extractor and classifier responses")
    parser.add_argument("--verbose", action="store_true", help="show library log messages")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    config = load_config(args)
    llm = ScriptedFakeLLM() if config.backend == "fake" else LLM(config.backend, config.model, config.base_url)
    if config.backend != "fake":
        llm.check()
    with tempfile.TemporaryDirectory() as tmp:
        mem = MemorySystem(config, llm, db_path=str(Path(tmp) / "demo_temporal.sqlite"), sample_id="demo")
        probe = Probe(mem, llm)
        header(f"TSOM temporal demo: backend={config.backend} model={config.model} "
               f"classifier={config.classifier_mode} "
               f"seed_multivalued={config.seed_multivalued} additive_cue={config.additive_cue}")
        run_steps(mem, llm, probe, args.raw)
        report(mem)
        answer_queries(mem, llm, config)
        mem.store.close()


def run_steps(mem, llm, probe, raw=False):
    seen_versions = 0
    for n, (statement, date, expected) in enumerate(STEPS, 1):
        probe.reset()
        mem.write(statement, date)

        header(f"STEP {n}/{len(STEPS)}")
        print(f"Input: {statement!r}   (session date {date})")
        print("\nAttribute names offered to the extractor: "
              + (", ".join(probe.hints[0]) if probe.hints and probe.hints[0] else "(none yet)"))
        print("\nFacts extracted (entity | attribute | value | valid_from | valid_until | tense):")
        for f in probe.extracted:
            print(f"  {f['entity']} | {f['attribute']} | {f['value']} | {f['valid_from'] or 'NONE'} | "
                  f"{f['valid_until'] or 'NONE'} | {f['temporal_status']}")
        if not probe.extracted:
            print("  (none)")

        print("\nOperations:")
        for d in probe.decisions:
            verdict = "as expected" if d["applied"] == expected else f"DIFFERS from expected {expected}"
            print(f"  {d['new']['entity']} | {d['new']['attribute']} | {d['new']['value']}"
                  f"  ->  {d['applied']}  ({verdict})")
            by = "LLM" if d["source"] == "LLM" else f"rule {d['source']} (no LLM call)"
            print(f"    decided by: {by}; {d['reason']}")
            if d["new"].get("raw_attribute"):
                print(f"    attribute snapped: extractor wrote {d['new']['raw_attribute']!r}, "
                      f"stored as {d['new']['attribute']!r}")
            if d["applied"] != d["op"]:
                print(f"    classifier said {d['op']}; the consolidator guard applied {d['applied']}")
            print("    touched: " + ("; ".join(describe(f) for f in d["old"]) if d["old"] else "no old facts"))
        if not probe.decisions:
            print(f"  (no fact was classified; expected {expected})")

        if raw:
            print("\nRaw extractor response:")
            for text in probe.raw_extractions or [None]:
                print("\n".join("  " + line for line in
                                 (text.strip() or "(empty)").splitlines()) if text is not None
                      else "  (extractor not called)")
            print("Raw classifier response:")
            shown = [d for d in probe.decisions if d["raw"] is not None]
            for d in shown:
                print("\n".join("  " + line for line in (d["raw"].strip() or "(empty)").splitlines()))
            if not shown:
                print("  (classifier did not call the LLM)")

        rows = mem.store.conn.execute(
            "SELECT * FROM note_versions WHERE rowid > ? ORDER BY rowid", (seen_versions,)).fetchall()
        seen_versions += len(rows)
        print("\nNote versions created by A-Mem evolution (old state):")
        for r in rows:
            print(f"  note {short(r['note_id'])} v{r['version']}  context={r['context']!r}  "
                  f"tags={json.loads(r['tags_json'])}  changed by note {short(r['changed_by_note_id'])}")
        if not rows:
            print("  none")

        write_op, temporal_op = mem.store.ops()[-2:]
        print(f"\nLLM calls for this write: {write_op.llm_calls + temporal_op.llm_calls} "
              f"(A-Mem {write_op.llm_calls}, temporal {temporal_op.llm_calls})"
              + (f"; {temporal_op.failures} temporal step(s) failed and were skipped" if temporal_op.failures else ""))


def report(mem):
    header("FACTS TABLE (ordered by entity, attribute, observed_at)")
    rows = mem.store.conn.execute(
        "SELECT * FROM facts ORDER BY entity_key, attr_key, observed_at, rowid").fetchall()
    print(table(
        ["id", "entity", "attribute", "value", "obs", "status", "operation",
         "valid_from", "valid_until", "supersedes", "superseded_by", "seen"],
        [[short(r["fact_id"]), r["entity"], r["attribute"], r["value"], r["observed_at"], r["status"],
          r["operation"], r["valid_from"] or "-", r["valid_until"] or "-",
          ",".join(short(i) for i in json.loads(r["supersedes"])) or "-",
          short(r["superseded_by"]), r["observation_count"]] for r in rows]))
    header("OPERATION COUNTS")
    counts = dict(mem.classifier.op_counts)
    print(", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "none")
    failures = sum(op.failures for op in mem.store.ops() if op.op_type == "temporal_write")
    print(f"temporal steps that failed and were skipped: {failures}")


def answer_queries(mem, llm, config):
    header("QUERIES (MemorySystem.search, then the LLM answers from that context)")
    for question, want, forbid in QUERIES:
        result = mem.search(question, config.retrieve_k)
        query_type, target = classify_query(question)
        prompt = LIST_ANSWER_PROMPT.format(context=result["notes"], question=question)
        answer = clean_prediction(llm.get_completion(prompt, temperature=0.0))
        low = answer.lower()
        ok = all(w.lower() in low for w in want) and not any(w.lower() in low for w in forbid)
        print(f"\nQ: {question}")
        print(f"query_type: {result['query_type']}" + (f" (target {target})" if target else ""))
        print(f"{FACTS_HEADER}\n" + (result["facts"] or "(no facts)"))
        print(f"Answer:   {answer!r}")
        print(f"Expected: {' and '.join(want)}" + (f", not {', '.join(forbid)}" if forbid else "")
              + "   ->   " + ("n/a (placeholder answer)" if config.backend == "fake"
                              else "OK" if ok else "DIFFERS"))
    if config.backend == "fake":
        print("\nNote: fake backend, so the answers above are placeholders.")


if __name__ == "__main__":
    main()
