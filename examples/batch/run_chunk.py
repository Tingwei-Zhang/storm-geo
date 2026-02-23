"""
Process a chunk of manifest rows: run Co-STORM for each, write failure summary, upload to S3.

Reads manifest (local or S3), slices by --start-index/--end-index or --chunk-id + chunk size,
runs run_single_query for each row (skip if output exists), writes chunk_summary.json
with success/failure counts and failed question_ids, then uploads output_dir to S3.
"""

from __future__ import annotations

import csv
import faulthandler
import json
import re
import sys
import tempfile
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

# Ensure stderr is unbuffered so errors appear immediately and enable faulthandler
sys.stderr.reconfigure(line_buffering=True)
faulthandler.enable()  # dump Python traceback on fatal signals (e.g., segfault)

# Optional S3
try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

from .run_single_query import load_ugc_replacement_map, run_single_query


def load_manifest_rows(manifest_path: Path | None, manifest_s3_uri: str | None) -> list[dict]:
    """Load manifest as list of dicts from local path or S3."""
    if manifest_s3_uri:
        if not HAS_BOTO3:
            raise RuntimeError("S3 requires boto3. Install with: pip install boto3")
        match = re.match(r"s3://([^/]+)/(.+)$", manifest_s3_uri.strip())
        if not match:
            raise ValueError(f"Invalid S3 URI: {manifest_s3_uri}")
        bucket, key = match.group(1), match.group(2)
        client = boto3.client("s3")
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            try:
                client.download_fileobj(bucket, key, f)
                f.flush()
                f.close()
                path = Path(f.name)
                rows = _read_manifest_csv(path)
            finally:
                path.unlink(missing_ok=True)
        return rows
    if manifest_path and manifest_path.exists():
        return _read_manifest_csv(manifest_path)
    raise FileNotFoundError("Provide --manifest or --manifest-s3 with a valid path/URI.")


def _read_manifest_csv(path: Path) -> list[dict]:
    """Read manifest CSV into list of dicts (normalize keys)."""
    rows = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): v for k, v in row.items()})
    return rows


def upload_dir_to_s3(local_dir: Path, s3_uri_prefix: str) -> None:
    """Upload a directory tree to S3 under the given prefix (e.g. s3://bucket/batch/results/chunk-0/)."""
    if not HAS_BOTO3:
        raise RuntimeError("S3 upload requires boto3. Install with: pip install boto3")
    match = re.match(r"s3://([^/]+)/(.*)$", s3_uri_prefix.rstrip("/"))
    if not match:
        raise ValueError(f"Invalid S3 prefix: {s3_uri_prefix}")
    bucket = match.group(1)
    prefix = (match.group(2) + "/").lstrip("/")
    client = boto3.client("s3")
    local_dir = Path(local_dir).resolve()
    for f in local_dir.rglob("*"):
        if f.is_file():
            key = prefix + f.relative_to(local_dir).as_posix()
            client.upload_file(str(f), bucket, key)


def run_chunk(
    rows: list[dict],
    output_dir: Path,
    *,
    injection_base_dir: Path | None = None,
    ugc_injection_mode: str = "replace_all",
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
    Run Co-STORM for each row in rows. Returns summary with success_count, failure_count, failed.
    """
    failed = []
    success_count = 0
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        qid = (row.get("question_id") or "").strip()
        topic = (row.get("topic") or "").strip()
        if not qid or not topic:
            failed.append({"question_id": qid, "error": "missing question_id or topic"})
            continue
        injection_path = (row.get("injection_doc_path") or "").strip() or None
        injection_s3 = (row.get("injection_doc_s3_uri") or "").strip() or None
        url_replacement_map = None
        ugc_path_str = (row.get("ugc_injection_path") or "").strip()
        if ugc_path_str:
            ugc_path = Path(ugc_path_str).expanduser()
            if not ugc_path.is_absolute() and injection_base_dir is not None:
                ugc_path = (injection_base_dir / ugc_path).resolve()
            else:
                ugc_path = ugc_path.resolve()
            if ugc_path.exists():
                url_replacement_map = load_ugc_replacement_map(ugc_path)
        try:
            run_single_query(
                question_id=qid,
                topic=topic,
                output_dir=output_dir,
                injection_doc_path=injection_path,
                injection_doc_s3_uri=injection_s3,
                url_replacement_map=url_replacement_map,
                url_replacement_first_only=(ugc_injection_mode == "first_only"),
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
            success_count += 1
        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"ERROR processing {qid}: {error_msg}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            failed.append({"question_id": qid, "error": error_msg})

    summary = {
        "total": len(rows),
        "success_count": success_count,
        "failure_count": len(failed),
        "failed": failed,
    }
    summary_path = output_dir / "chunk_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    print("Starting run_chunk...", file=sys.stderr)
    parser = ArgumentParser(
        description="Run a chunk of manifest rows through Co-STORM and upload to S3."
    )
    parser.add_argument("--manifest", type=Path, default=None, help="Local manifest CSV.")
    parser.add_argument("--manifest-s3", type=str, default=None, help="S3 URI of manifest CSV.")
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Start row index (inclusive).",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="End row index (exclusive).",
    )
    parser.add_argument(
        "--chunk-id",
        type=int,
        default=None,
        help="Chunk index (use with --chunk-size to compute start/end).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Rows per chunk when using --chunk-id.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--injection-base-dir",
        type=Path,
        default=None,
        help="Base dir to resolve relative ugc_injection_path (e.g. when manifest is from S3).",
    )
    parser.add_argument(
        "--ugc-injection-mode",
        type=str,
        choices=["replace_all", "first_only"],
        default="replace_all",
        help="replace_all: replace every matching URL. first_only: general UGC mode, replace only the first matching UGC result.",
    )
    parser.add_argument(
        "--upload-s3",
        type=str,
        default=None,
        help="Upload output_dir to this S3 prefix (e.g. s3://bucket/results/chunk-0/).",
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

    if args.chunk_id is not None:
        start = args.chunk_id * args.chunk_size
        end = min(start + args.chunk_size, len(rows))
        rows = rows[start:end]
    elif args.start_index is not None and args.end_index is not None:
        rows = rows[args.start_index : args.end_index]
    # else: use all rows

    if not rows:
        print("No rows in chunk.", file=sys.stderr)
        return 0

    injection_base_dir = args.injection_base_dir
    if injection_base_dir is None and args.manifest is not None:
        injection_base_dir = args.manifest.resolve().parent

    print(f"Processing {len(rows)} rows in chunk...", file=sys.stderr)
    summary = run_chunk(
        rows,
        args.output_dir,
        injection_base_dir=injection_base_dir,
        ugc_injection_mode=args.ugc_injection_mode,
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
        f"Chunk done: {summary['success_count']} ok, {summary['failure_count']} failed"
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
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        print(f"FATAL ERROR in run_chunk: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
