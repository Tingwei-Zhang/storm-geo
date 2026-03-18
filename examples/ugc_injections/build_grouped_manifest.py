"""
Build a manifest from grouped_ugc_clusters.csv for the _grouped pipeline.

Reads grouped_ugc_clusters.csv (group_name, query_template, filled_query,
ugc_url, ugc_domain, cluster_id, ...), filters by selected group names, and
writes a manifest CSV with question_id, topic (= filled_query), and optional
ugc_injection_path for use with run_local_parallel.

Usage:
  # Base experiment (no injections): manifest with empty ugc_injection_path
  python -m examples.ugc_injections.build_grouped_manifest \\
    --grouped-csv /path/to/grouped_ugc_clusters.csv \\
    --group-names "401k early withdrawal" "AAA alternative" \\
    --output-dir experiment_config/base_grouped

  # With injections: same manifest but ugc_injection_path = injection_{group}.json
  python -m examples.ugc_injections.build_grouped_manifest \\
    --grouped-csv /path/to/grouped_ugc_clusters.csv \\
    --group-names "401k early withdrawal" "AAA alternative" \\
    --output-dir experiment_config/classic_grouped --with-injection-path
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


def _sanitize_group_name(name: str) -> str:
    """Safe filename segment from group name."""
    s = (name or "").strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[-\s]+", "_", s).strip("_")
    return s.lower() or "group"


def load_grouped_csv(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): v for k, v in row.items()})
    return rows


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Build manifest from grouped_ugc_clusters.csv for selected group names."
    )
    parser.add_argument(
        "--grouped-csv",
        type=Path,
        required=True,
        help="Path to grouped_ugc_clusters.csv.",
    )
    parser.add_argument(
        "--group-names",
        nargs="+",
        required=True,
        help="Group names to include (e.g. '401k early withdrawal' 'AAA alternative').",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory (e.g. experiment_config/base_grouped).",
    )
    parser.add_argument(
        "--with-injection-path",
        action="store_true",
        help="Set ugc_injection_path to injection_{group}.json per row.",
    )
    args = parser.parse_args()

    if not args.grouped_csv.exists():
        print(f"Grouped CSV not found: {args.grouped_csv}", file=sys.stderr)
        return 1

    wanted = {g.strip() for g in args.group_names if g.strip()}
    if not wanted:
        print("Provide at least one --group-names.", file=sys.stderr)
        return 1

    rows = load_grouped_csv(args.grouped_csv)
    seen_filled: set[str] = set()
    filtered = []
    for r in rows:
        gn = (r.get("group_name") or "").strip()
        if gn not in wanted:
            continue
        filled = (r.get("filled_query") or "").strip()
        if not filled or filled in seen_filled:
            continue
        seen_filled.add(filled)
        filtered.append(r)

    if not filtered:
        print("No rows found for the given group names.", file=sys.stderr)
        return 1

    # Build manifest rows: question_id, topic required by run_local_parallel
    manifest_rows = []
    for i, r in enumerate(filtered):
        gn = (r.get("group_name") or "").strip()
        filled = (r.get("filled_query") or "").strip()
        safe = _sanitize_group_name(gn)
        qid = f"grouped_{safe}_{i}"
        inj_path = ""
        if args.with_injection_path:
            inj_path = f"injection_{safe}.json"
        manifest_rows.append({
            "question_id": qid,
            "original_question_id": str(i),
            "dataset": "grouped",
            "topic": filled,
            "title": filled,
            "body": r.get("body") or "",
            "domain": (r.get("ugc_domain") or "").strip(),
            "cluster_id": (r.get("cluster_id") or "").strip(),
            "injection_doc_path": "",
            "injection_doc_s3_uri": "",
            "ugc_injection_path": inj_path,
            "group_name": gn,
            "query_template": (r.get("query_template") or "").strip(),
            "ugc_url": (r.get("ugc_url") or "").strip(),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / "manifest.csv"
    fieldnames = [
        "question_id", "original_question_id", "dataset", "topic", "title", "body",
        "domain", "cluster_id", "injection_doc_path", "injection_doc_s3_uri",
        "ugc_injection_path", "group_name", "query_template", "ugc_url",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(manifest_rows)

    print(f"Wrote {len(manifest_rows)} rows to {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
