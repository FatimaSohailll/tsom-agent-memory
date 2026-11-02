"""Run one ablation suite and rebuild tables.md.

    python scripts/run_experiments.py --suite actions
    python scripts/run_experiments.py --suite scenarios --configs amem,tsom_both --limit 2
    python scripts/run_experiments.py --suite locomo --conversations 1 --out /content/drive/MyDrive/g1_results/ablation

Suites:
  actions    harness.eval_actions twice: LLM labels (gap1_llm) and newest value wins (gap1_lww).
             tsom_conflict and tsom_both share gap1_llm; tsom_time shares gap1_lww. --configs is ignored.
  scenarios  harness.run_scenarios for each config.
  locomo     harness.run_locomo on the first --conversations conversations for each config,
             or on exactly --conversation-ids (e.g. 1,8) if given (--limit caps questions
             per conversation).

Configs are names in configs/exp/ (amem, tsom_conflict, tsom_time, tsom_both), "rag"
for configs/rag_qwen2.5-3b.json, or a path to a .json file. Results go to
<out>/<suite>/<name>/; every run is resumable. --set overrides apply to every config.
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness import eval_actions, run_locomo, run_scenarios  # noqa: E402
from harness.make_tables import make_tables  # noqa: E402
from harness.run_locomo import load_config, parse_ids  # noqa: E402
from harness.scenarios import DEFAULT_SCENARIOS  # noqa: E402

DEFAULT_CONFIGS = "amem,tsom_conflict,tsom_time,tsom_both"


def config_path(name: str) -> Path:
    if name.endswith(".json"):
        return Path(name)
    if name == "rag":
        return ROOT / "configs" / "rag_qwen2.5-3b.json"
    return ROOT / "configs" / "exp" / f"{name}.json"


def config_name(name: str) -> str:
    return Path(name).stem if name.endswith(".json") else name


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True, choices=["actions", "scenarios", "locomo"])
    parser.add_argument("--configs", default=DEFAULT_CONFIGS)
    parser.add_argument("--conversations", type=int, default=1)
    parser.add_argument("--conversation-ids", type=parse_ids, metavar="IDS",
                        help="comma-separated LoCoMo indices (locomo suite); overrides --conversations")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", default="results/ablation")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--scenarios", default=DEFAULT_SCENARIOS)
    parser.add_argument("--include-unreviewed", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("amem_robust").setLevel(logging.WARNING)

    out = Path(args.out)
    suite_dir = out / args.suite
    if args.suite == "actions":
        for run_name, use_classifier in (("gap1_llm", True), ("gap1_lww", False)):
            config = load_config(config_path("tsom_both"), [
                *args.set, f"run_name={run_name}", f"output_dir={suite_dir}",
                f"use_classifier={'true' if use_classifier else 'false'}"])
            eval_actions.run(config, args.scenarios, args.limit, args.include_unreviewed)
    else:
        for name in [c.strip() for c in args.configs.split(",") if c.strip()]:
            overrides = [*args.set, f"run_name={config_name(name)}", f"output_dir={suite_dir}"]
            if args.suite == "scenarios":
                run_scenarios.run(load_config(config_path(name), overrides), args.scenarios, args.limit,
                                  args.include_unreviewed)
            else:
                if args.conversation_ids is None:
                    overrides.append(f"max_samples={args.conversations}")
                if args.limit is not None:
                    overrides.append(f"max_questions={args.limit}")
                run_locomo.run(load_config(config_path(name), overrides, args.conversation_ids))

    make_tables(out)
    print(f"tables written to {out / 'tables.md'}")


if __name__ == "__main__":
    main()
