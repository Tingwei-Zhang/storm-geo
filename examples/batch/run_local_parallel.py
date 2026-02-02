"""
Run Co-STORM over manifest rows in parallel using multiprocessing.

Reads manifest (local or S3), spawns a Pool of workers; each worker runs
run_single_query for a subset of rows. Same skip-if-exists and output layout.
Optionally sync output_dir to S3 at the end.
"""

from __future__ import annotations

import csv
import json
import multiprocessing as mp
import re
import sys
import tempfile
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Optional

try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

from .run_chunk import load_manifest_rows, upload_dir_to_s3
from .run_single_query import run_single_query


def _run_one(
    row: dict,
    output_dir: str,
    retriever: str,
    demo_turns: int,
    retrieve_top_k: int,
    total_conv_turn: int,
    max_search_queries: int,
    max_search_thread: int,
    max_search_queries_per_turn: int,
    warmstart_max_num_experts: int,
    warmstart_max_turn_per_experts: int,
    warmstart_max_thread: int,
    max_thread_num: int,
    max_num_round_table_experts: int,
    moderator_override_n: int,
    node_expansion_trigger_count: int,
    lm_preset: str,
    skip_if_exists: bool,
) -> tuple[str, Optional[str]]:
    """
    Run one row. Returns (question_id, error_message or None).
    Must be a top-level function for multiprocessing.Pool.
    """
    qid = (row.get("question_id") or "").strip()
    topic = (row.get("topic") or "").strip()
    if not qid or not topic:
        return (qid or "unknown", "missing question_id or topic")
    injection_path = (row.get("injection_doc_path") or "").strip() or None
    injection_s3 = (row.get("injection_doc_s3_uri") or "").strip() or None
    try:
        run_single_query(
            question_id=qid,
            topic=topic,
            output_dir=Path(output_dir),
            injection_doc_path=injection_path,
            injection_doc_s3_uri=injection_s3,
            retriever=retriever,
            demo_turns=demo_turns,
            retrieve_top_k=retrieve_top_k,
            total_conv_turn=total_conv_turn,
            max_search_queries=max_search_queries,
            max_search_thread=max_search_thread,
            max_search_queries_per_turn=max_search_queries_per_turn,
            warmstart_max_num_experts=warmstart_max_num_experts,
            warmstart_max_turn_per_experts=warmstart_max_turn_per_experts,
            warmstart_max_thread=warmstart_max_thread,
            max_thread_num=max_thread_num,
            max_num_round_table_experts=max_num_round_table_experts,
            moderator_override_N_consecutive_answering_turn=moderator_override_n,
            node_expansion_trigger_count=node_expansion_trigger_count,
            lm_preset=lm_preset,
            skip_if_exists=skip_if_exists,
        )
        return (qid, None)
    except Exception as e:
        return (qid, str(e))


def run_local_parallel(
    rows: list[dict],
    output_dir: Path,
    workers: int = 4,
    *,
    retriever: str = "google",
    demo_turns: int = 2,
    retrieve_top_k: int = 3,
    total_conv_turn: int = 20,
    max_search_queries: int = 2,
    max_search_thread: int = 5,
    max_search_queries_per_turn: int = 3,
    warmstart_max_num_experts: int = 1,
    warmstart_max_turn_per_experts: int = 1,
    warmstart_max_thread: int = 1,
    max_thread_num: int = 5,
    max_num_round_table_experts: int = 1,
    moderator_override_n: int = 2,
    node_expansion_trigger_count: int = 10,
    lm_preset: str = "demo",
    skip_if_exists: bool = True,
) -> dict[str, Any]:
    """
    Run Co-STORM for each row in parallel. Returns summary with success_count, failed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_str = str(output_dir.resolve())

    def _args_for_row(row: dict) -> tuple:
        return (
            row,
            out_str,
            retriever,
            demo_turns,
            retrieve_top_k,
            total_conv_turn,
            max_search_queries,
            max_search_thread,
            max_search_queries_per_turn,
            warmstart_max_num_experts,
            warmstart_max_turn_per_experts,
            warmstart_max_thread,
            max_thread_num,
            max_num_round_table_experts,
            moderator_override_n,
            node_expansion_trigger_count,
            lm_preset,
            skip_if_exists,
        )

    failed = []
    success_count = 0
    with mp.Pool(processes=workers) as pool:
        results = pool.starmap(_run_one, [_args_for_row(row) for row in rows])

    for qid, err in results:
        if err is not None:
            failed.append({"question_id": qid, "error": err})
        else:
            success_count += 1

    summary = {
        "total": len(rows),
        "success_count": success_count,
        "failure_count": len(failed),
        "failed": failed,
    }
    summary_path = output_dir / "parallel_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = ArgumentParser(
        description="Run Co-STORM over manifest in parallel (multiprocessing)."
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--manifest-s3", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--upload-s3",
        type=str,
        default=None,
        help="After run, sync output_dir to this S3 prefix.",
    )
    parser.add_argument("--retriever", type=str, default="google")
    parser.add_argument("--demo-turns", type=int, default=2)
    parser.add_argument("--retrieve-top-k", type=int, default=3)
    parser.add_argument("--total-conv-turn", type=int, default=20)
    parser.add_argument("--max-search-queries", type=int, default=2)
    parser.add_argument("--max-search-thread", type=int, default=5)
    parser.add_argument("--max-search-queries-per-turn", type=int, default=3)
    parser.add_argument("--warmstart-max-num-experts", type=int, default=1)
    parser.add_argument("--warmstart-max-turn-per-experts", type=int, default=1)
    parser.add_argument("--warmstart-max-thread", type=int, default=1)
    parser.add_argument("--max-thread-num", type=int, default=5)
    parser.add_argument("--max-num-round-table-experts", type=int, default=1)
    parser.add_argument("--moderator-override-n", type=int, default=2)
    parser.add_argument("--node-expansion-trigger-count", type=int, default=10)
    parser.add_argument("--lm-preset", type=str, choices=["demo", "gpt"], default="demo")
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    if (args.manifest is None) == (args.manifest_s3 is None):
        print("Provide exactly one of --manifest or --manifest-s3.", file=sys.stderr)
        return 1

    rows = load_manifest_rows(args.manifest, args.manifest_s3)
    if not rows:
        print("No rows in manifest.", file=sys.stderr)
        return 0

    summary = run_local_parallel(
        rows,
        args.output_dir,
        workers=args.workers,
        retriever=args.retriever,
        demo_turns=args.demo_turns,
        retrieve_top_k=args.retrieve_top_k,
        total_conv_turn=args.total_conv_turn,
        max_search_queries=args.max_search_queries,
        max_search_thread=args.max_search_thread,
        max_search_queries_per_turn=args.max_search_queries_per_turn,
        warmstart_max_num_experts=args.warmstart_max_num_experts,
        warmstart_max_turn_per_experts=args.warmstart_max_turn_per_experts,
        warmstart_max_thread=args.warmstart_max_thread,
        max_thread_num=args.max_thread_num,
        max_num_round_table_experts=args.max_num_round_table_experts,
        moderator_override_n=args.moderator_override_n,
        node_expansion_trigger_count=args.node_expansion_trigger_count,
        lm_preset=args.lm_preset,
        skip_if_exists=not args.no_skip_existing,
    )

    print(
        f"Done: {summary['success_count']} ok, {summary['failure_count']} failed"
    )
    if summary["failed"]:
        for item in summary["failed"][:10]:
            print(f"  failed {item['question_id']}: {item['error']}")
        if len(summary["failed"]) > 10:
            print(f"  ... and {len(summary['failed']) - 10} more")

    if args.upload_s3:
        upload_dir_to_s3(args.output_dir, args.upload_s3)
        print(f"Uploaded to {args.upload_s3}")

    return 0 if summary["failure_count"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
