"""Small stdlib CLI for local use and VS Code tasks."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from devassist.config import Settings
from devassist.evaluation.runner import evaluate, write_report
from devassist.ingestion import ChunkingConfig, ingest_jsonl
from devassist.models import DiagnoseRequest, DiagnoseResponse, SearchHit, SearchRequest
from devassist.service import DevAssistService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="devassist")
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="Run the FastAPI service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    search = commands.add_parser("search", help="Inspect retrieval results")
    search.add_argument("query")
    search.add_argument("--project")
    search.add_argument("--version")
    search.add_argument("--top-k", type=int)
    search.add_argument(
        "--mode",
        choices=sorted({"bm25", "dense", "hybrid", "hybrid_rerank"}),
        default="hybrid_rerank",
    )

    diagnose = commands.add_parser("diagnose", help="Run the bounded support agent")
    diagnose.add_argument("query")
    diagnose.add_argument("--project")
    diagnose.add_argument("--version")
    diagnose.add_argument("--top-k", type=int)

    evaluation = commands.add_parser("evaluate", help="Run offline evaluation")
    evaluation.add_argument("--output", type=Path, default=Path("reports/evaluation.json"))

    ingest = commands.add_parser("ingest", help="Clean and chunk a JSONL corpus")
    ingest.add_argument("source", type=Path)
    ingest.add_argument("destination", type=Path)
    ingest.add_argument("--max-chars", type=int, default=1_000)
    ingest.add_argument("--overlap-chars", type=int, default=150)
    return parser


def serialise_output(output: DiagnoseResponse | list[SearchHit]) -> str:
    if isinstance(output, list):
        return json.dumps(
            [item.model_dump(mode="json") for item in output],
            ensure_ascii=False,
            indent=2,
        )
    return output.model_dump_json(indent=2)


def main() -> None:
    args = build_parser().parse_args()
    settings = Settings.from_env()
    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "devassist.api.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
        return
    if args.command == "evaluate":
        report = evaluate(settings)
        output = args.output if args.output.is_absolute() else settings.root / args.output
        write_report(report, output)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.command == "ingest":
        result = ingest_jsonl(
            args.source,
            args.destination,
            chunking=ChunkingConfig(max_chars=args.max_chars, overlap_chars=args.overlap_chars),
        )
        print(json.dumps(asdict(result.stats), ensure_ascii=False, indent=2))
        return

    service = DevAssistService(settings)
    if args.command == "search":
        search_values = {
            "query": args.query,
            "project": args.project,
            "version": args.version,
            "mode": args.mode,
        }
        if args.top_k is not None:
            search_values["top_k"] = args.top_k
        output = service.search(SearchRequest.model_validate(search_values))
    else:
        diagnose_values = {
            "query": args.query,
            "project": args.project,
            "version": args.version,
        }
        if args.top_k is not None:
            diagnose_values["top_k"] = args.top_k
        output = service.diagnose(DiagnoseRequest.model_validate(diagnose_values))
    print(serialise_output(output))


if __name__ == "__main__":
    main()
