"""
GEO local experiments with randomized injection location.

This script keeps `run_geo_experiment_local.py` unchanged and provides a variant
that samples:
- m: retrieval call number where the snippet is injected (1-based)
- n: document insertion index within that retrieval result list (0-based)

Sampling can be deterministic with `--injection-seed`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing as mp
import random
import sys
from argparse import ArgumentParser
from pathlib import Path

try:
    from examples.batch.run_chunk import load_manifest_rows
    from examples.batch.run_single_query import run_single_query
    from examples.geo_examples.run_geo_experiment_local import (
        _normalize_geo_snippet_dict_to_retrieved_format,
        _normalize_geo_snippet_to_retrieved_format,
    )
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from examples.batch.run_chunk import load_manifest_rows  # type: ignore[reportMissingImports]
    from examples.batch.run_single_query import run_single_query  # type: ignore[reportMissingImports]
    from examples.geo_examples.run_geo_experiment_local import (  # type: ignore[reportMissingImports]
        _normalize_geo_snippet_dict_to_retrieved_format,
        _normalize_geo_snippet_to_retrieved_format,
    )


def _derive_seed(base_seed: int, run_question_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{run_question_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _pick_injection_targets(
    *,
    run_question_id: str,
    randomize_m: bool,
    m_min: int,
    m_max: int,
    randomize_n: bool,
    n_min: int,
    n_max: int,
    base_seed: int | None,
) -> tuple[int, int]:
    if base_seed is None:
        rng: random.Random = random.SystemRandom()
    else:
        rng = random.Random(_derive_seed(base_seed, run_question_id))

    target_m = rng.randint(m_min, m_max) if randomize_m else m_min
    target_n = rng.randint(n_min, n_max) if randomize_n else n_min
    return target_m, target_n


def _run_one(
    question_id: str,
    topic: str,
    output_dir: Path,
    injection_doc_path: str | None,
    kwargs: dict,
    injection_retrieval_number: int,
    injection_position: int,
) -> bool:
    return run_single_query(
        question_id=question_id,
        topic=topic,
        output_dir=output_dir,
        injection_doc_path=injection_doc_path,
        injection_doc_s3_uri=None,
        injection_retrieval_number=injection_retrieval_number,
        injection_position=injection_position,
        **kwargs,
    )


def _run_one_item(
    item: dict,
    output_dir: Path,
    default_geo_path: str | None,
    kwargs: dict,
) -> tuple[str, str, bool, str]:
    qid = item["run_question_id"]
    topic = item["topic"]
    geo_id = item.get("geo_id", "")
    per_row_geo_doc = item.get("geo_doc")
    target_m = item["injection_retrieval_number"]
    target_n = item["injection_position"]

    row_geo_temp: Path | None = None
    try:
        row_geo_path: str | None = None
        if per_row_geo_doc is not None:
            normalized_path, row_geo_temp = _normalize_geo_snippet_dict_to_retrieved_format(
                per_row_geo_doc, source_label=f"geo_id={geo_id}"
            )
            row_geo_path = str(normalized_path)

        effective_geo_path = row_geo_path if row_geo_path is not None else default_geo_path
        _run_one(
            qid,
            topic,
            output_dir,
            effective_geo_path,
            kwargs,
            injection_retrieval_number=target_m,
            injection_position=target_n,
        )
        return (qid, geo_id, True, "")
    except Exception as e:
        return (qid, geo_id, False, str(e))
    finally:
        if row_geo_temp is not None and row_geo_temp.exists():
            row_geo_temp.unlink(missing_ok=True)


def _run_one_item_star(args: tuple[dict, Path, str | None, dict]) -> tuple[str, str, bool, str]:
    return _run_one_item(*args)


def main() -> int:
    parser = ArgumentParser(
        description="GEO experiment with randomized injection retrieval round (m) and doc index (n)."
    )
    parser.add_argument("--question-id", type=str, default=None)
    parser.add_argument("--topic", type=str, default=None)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Local manifest CSV (question_id, topic).",
    )
    parser.add_argument(
        "--geo-dataset",
        type=Path,
        default=None,
        help="CSV from build_geo_dataset.py with geo_id, question_id, query, and geo_document (JSON string).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for run artifacts. If omitted with --geo-dataset, "
            "defaults to <geo-dataset-without-.csv>_runs."
        ),
    )
    parser.add_argument(
        "--geo-snippet",
        type=Path,
        default=None,
        help="Path to one webset snippet JSON. Same format as manual_document_example.json.",
    )

    parser.add_argument("--randomize-m", dest="randomize_m", action="store_true", default=True)
    parser.add_argument("--no-randomize-m", dest="randomize_m", action="store_false")
    parser.add_argument("--randomize-n", dest="randomize_n", action="store_true", default=True)
    parser.add_argument("--no-randomize-n", dest="randomize_n", action="store_false")
    parser.add_argument("--m-min", type=int, default=1, help="Minimum retrieval call index for injection (1-based).")
    parser.add_argument("--m-max", type=int, default=3, help="Maximum retrieval call index for injection (1-based).")
    parser.add_argument("--n-min", type=int, default=0, help="Minimum insertion index in retrieval result list (0-based).")
    parser.add_argument(
        "--n-max",
        type=int,
        default=None,
        help="Maximum insertion index (0-based). Defaults to --retrieve-top-k.",
    )
    parser.add_argument("--injection-seed", type=int, default=None, help="Optional seed for deterministic m/n sampling.")

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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for running multiple queries (default: 1).",
    )
    args = parser.parse_args()

    mode_count = int(args.geo_dataset is not None) + int(args.manifest is not None) + int(
        bool(args.question_id and args.topic)
    )
    if mode_count != 1:
        print(
            "Choose exactly one input mode: (--geo-dataset) OR (--manifest) OR (--question-id and --topic).",
            file=sys.stderr,
        )
        return 1

    if args.geo_dataset is None and args.geo_snippet is None:
        print("Injection mode requires --geo-snippet (or use --geo-dataset).", file=sys.stderr)
        return 1

    if args.m_min < 1 or args.m_max < 1 or args.m_min > args.m_max:
        print("Invalid m range: require 1 <= m-min <= m-max.", file=sys.stderr)
        return 1

    n_max = args.retrieve_top_k if args.n_max is None else args.n_max
    if args.n_min < 0 or n_max < 0 or args.n_min > n_max:
        print("Invalid n range: require 0 <= n-min <= n-max.", file=sys.stderr)
        return 1

    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    elif args.geo_dataset is not None:
        output_dir = Path(f"{args.geo_dataset.with_suffix('')}_runs").resolve()
        print(f"Auto output-dir from --geo-dataset: {output_dir}")
    else:
        print(
            "--output-dir is required unless --geo-dataset is provided.",
            file=sys.stderr,
        )
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    geo_path: str | None = None
    geo_temp_path: Path | None = None
    if args.geo_snippet is not None:
        if not args.geo_snippet.exists():
            print(f"Geo snippet not found: {args.geo_snippet}", file=sys.stderr)
            return 1
        try:
            normalized_path, geo_temp_path = _normalize_geo_snippet_to_retrieved_format(args.geo_snippet)
            geo_path = str(normalized_path)
            print(f"GEO snippet source: {args.geo_snippet}")
            print("  Normalized to retrieved-doc format: url, title, description, snippets.")
        except (ValueError, json.JSONDecodeError) as e:
            print(f"Geo snippet invalid: {e}", file=sys.stderr)
            return 1

    def _cleanup_and_return(code: int) -> int:
        if geo_temp_path is not None and geo_temp_path.exists():
            geo_temp_path.unlink(missing_ok=True)
        return code

    common_kw = dict(
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
        moderator_override_N_consecutive_answering_turn=args.moderator_override_n,
        node_expansion_trigger_count=args.node_expansion_trigger_count,
        lm_preset=args.lm_preset,
        skip_if_exists=not args.no_skip_existing,
    )

    def gather_queries():
        if args.geo_dataset is not None:
            if not args.geo_dataset.exists():
                print(f"GEO dataset not found: {args.geo_dataset}", file=sys.stderr)
                return None
            print(f"Loading GEO dataset from {args.geo_dataset} ...")
            with args.geo_dataset.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            print(f"  Loaded {len(rows)} rows.")
            out = []
            for r in rows:
                geo_id = (r.get("geo_id") or "").strip()
                qid = (r.get("question_id") or "").strip()
                query = (r.get("query") or "").strip()
                geo_doc_text = (r.get("geo_document") or "").strip()
                if not geo_id or not qid or not query or not geo_doc_text:
                    continue
                try:
                    geo_doc = json.loads(geo_doc_text)
                except json.JSONDecodeError:
                    print(f"  Skip row geo_id={geo_id}: invalid geo_document JSON.", file=sys.stderr)
                    continue
                out.append(
                    {
                        "run_question_id": f"{geo_id}__{qid}",
                        "topic": query,
                        "geo_id": geo_id,
                        "geo_doc": geo_doc,
                    }
                )
            return out

        if args.manifest is not None:
            if not args.manifest.exists():
                print(f"Manifest not found: {args.manifest}", file=sys.stderr)
                return None
            print(f"Loading manifest from {args.manifest} ...")
            rows = load_manifest_rows(args.manifest, None)
            print(f"  Loaded {len(rows)} rows.")
            return [
                {
                    "run_question_id": (r.get("question_id") or "").strip(),
                    "topic": (r.get("topic") or "").strip(),
                    "geo_id": "",
                    "geo_doc": None,
                }
                for r in rows
                if (r.get("question_id") or "").strip() and (r.get("topic") or "").strip()
            ]
        if args.question_id and args.topic:
            return [
                {
                    "run_question_id": args.question_id,
                    "topic": args.topic,
                    "geo_id": "",
                    "geo_doc": None,
                }
            ]
        print("Provide one input mode: --geo-dataset, --manifest, or --question-id/--topic.", file=sys.stderr)
        return None

    queries = gather_queries()
    if queries is None:
        return _cleanup_and_return(1)
    if not queries:
        print("No queries to run.", file=sys.stderr)
        return _cleanup_and_return(0)

    print(
        "Injection sampling config: "
        f"randomize_m={args.randomize_m}, m_range=[{args.m_min},{args.m_max}], "
        f"randomize_n={args.randomize_n}, n_range=[{args.n_min},{n_max}], "
        f"seed={args.injection_seed}"
    )
    for item in queries:
        target_m, target_n = _pick_injection_targets(
            run_question_id=item["run_question_id"],
            randomize_m=args.randomize_m,
            m_min=args.m_min,
            m_max=args.m_max,
            randomize_n=args.randomize_n,
            n_min=args.n_min,
            n_max=n_max,
            base_seed=args.injection_seed,
        )
        item["injection_retrieval_number"] = target_m
        item["injection_position"] = target_n

    workers = max(1, args.workers)
    total = len(queries)
    if workers > 1 and total > 1:
        print(f"Running {total} queries in parallel with workers={workers} ...")
        tasks = [(item, output_dir, geo_path, common_kw) for item in queries]
        with mp.Pool(processes=workers) as pool:
            for i, (qid, geo_id, ok, err) in enumerate(pool.imap_unordered(_run_one_item_star, tasks), start=1):
                prefix = f"  [{i}/{total}]"
                print(f"{prefix} question_id={qid!r} ...")
                if geo_id:
                    print(f"{prefix}   geo_id={geo_id}")
                matched = next((x for x in queries if x["run_question_id"] == qid), None)
                if matched is not None:
                    print(
                        f"{prefix}   injection_target=(m={matched['injection_retrieval_number']}, "
                        f"n={matched['injection_position']})"
                    )
                if ok:
                    print(f"{prefix}   Done -> {output_dir}/")
                else:
                    print(f"{prefix}   Failed: {err}", file=sys.stderr)
    else:
        for i, item in enumerate(queries):
            qid = item["run_question_id"]
            topic = item["topic"]
            prefix = f"  [{i + 1}/{total}]" if total > 1 else "  "
            print(f"{prefix} question_id={qid!r} topic={topic[:50]!r}...")
            if item.get("geo_id"):
                print(f"{prefix}   geo_id={item['geo_id']}")
            print(
                f"{prefix}   injection_target=(m={item['injection_retrieval_number']}, "
                f"n={item['injection_position']})"
            )
            qid_out, geo_id_out, ok, err = _run_one_item(item, output_dir, geo_path, common_kw)
            if ok:
                print(f"{prefix}   Done -> {output_dir}/")
            else:
                if geo_id_out:
                    print(f"{prefix}   geo_id={geo_id_out}")
                print(f"{prefix}   Failed: {err}", file=sys.stderr)

    return _cleanup_and_return(0)


if __name__ == "__main__":
    sys.exit(main())
