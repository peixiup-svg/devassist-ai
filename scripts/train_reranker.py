"""Train blended local reranker weights from committed labelled pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from devassist.training import train_reranker  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge", type=Path, default=ROOT / "data/sample/knowledge_base.jsonl")
    parser.add_argument("--pairs", type=Path, default=ROOT / "data/training/reranker_pairs.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/reranker/weights.json")
    parser.add_argument("--blend", type=float, default=0.30)
    args = parser.parse_args()
    payload = train_reranker(
        knowledge_path=args.knowledge,
        pairs_path=args.pairs,
        output_path=args.output,
        blend=args.blend,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
