"""Repository-local entry point for the deterministic evaluation suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from devassist.config import Settings  # noqa: E402
from devassist.evaluation.runner import evaluate, write_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DevAssist AI offline.")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "evaluation.json")
    parser.add_argument("--fail-on-threshold", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = Settings.from_env(ROOT)
    report = evaluate(settings)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    write_report(report, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_threshold and not report["thresholds"]["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
