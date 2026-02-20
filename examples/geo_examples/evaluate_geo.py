from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


TARGET_NAMES = ("longevion", "brightcase", "bananacoin")


def _contains_target_name(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_target_name(v) for v in value.values())
    if isinstance(value, list):
        return any(_contains_target_name(v) for v in value)
    if isinstance(value, str):
        text = value.lower()
        return any(name in text for name in TARGET_NAMES)
    return False


def compute_metrics_for_run(run_dir: Path) -> tuple[bool, bool]:
    instance_dump_path = run_dir / "instance_dump.json"
    report_path = run_dir / "report.md"

    cited = False
    if instance_dump_path.exists():
        try:
            payload = json.loads(instance_dump_path.read_text(encoding="utf-8"))
            # Current run format stores citations under knowledge_base.info_uuid_to_info_dict.
            # Keep a top-level fallback for compatibility with older dumps.
            info_dict = (
                payload.get("knowledge_base", {}).get("info_uuid_to_info_dict")
                or payload.get("info_uuid_to_info_dict", {})
            )
            cited = _contains_target_name(info_dict)
        except (json.JSONDecodeError, OSError):
            cited = False

    mentioned = False
    if report_path.exists():
        try:
            report_text = report_path.read_text(encoding="utf-8").lower()
            mentioned = any(name in report_text for name in TARGET_NAMES)
        except OSError:
            mentioned = False

    return cited, mentioned


def _rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0/0 (0.0%)"
    return f"{numerator}/{denominator} ({(100.0 * numerator / denominator):.1f}%)"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate GEO run outputs for target-name presence in citations and report text."
        )
    )
    parser.add_argument(
        "dataset_csv",
        type=Path,
        help=(
            "Input dataset CSV path. "
            "Derived paths: <name>_runs for run directory and "
            "<name>_evaluated.csv for output."
        ),
    )
    args = parser.parse_args()

    dataset_csv = args.dataset_csv
    base_no_suffix = dataset_csv.with_suffix("")
    runs_dir = Path(f"{base_no_suffix}_runs")
    output_csv = Path(f"{base_no_suffix}_evaluated.csv")

    if not dataset_csv.exists():
        raise FileNotFoundError(f"Input dataset CSV not found: {dataset_csv}")
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    with dataset_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        input_rows = list(reader)
        input_fieldnames = list(reader.fieldnames or [])

    output_fieldnames = [*input_fieldnames, "cited", "mentioned"]
    output_rows: list[dict[str, str]] = []

    generated_reports = 0
    missing_run_dirs = 0
    by_domain_total: dict[str, int] = defaultdict(int)
    by_domain_report: dict[str, int] = defaultdict(int)
    by_domain_cited: dict[str, int] = defaultdict(int)
    by_domain_mentioned: dict[str, int] = defaultdict(int)
    by_geo_method_total: dict[str, int] = defaultdict(int)
    by_geo_method_cited: dict[str, int] = defaultdict(int)
    by_geo_method_mentioned: dict[str, int] = defaultdict(int)
    by_goal_type_total: dict[str, int] = defaultdict(int)
    by_goal_type_cited: dict[str, int] = defaultdict(int)
    by_goal_type_mentioned: dict[str, int] = defaultdict(int)

    for row in input_rows:
        geo_id = (row.get("geo_id") or "").strip()
        question_id = (row.get("question_id") or "").strip()
        domain = question_id.split("_", maxsplit=1)[0] if question_id else "unknown"
        geo_method = (row.get("geo_method") or "").strip() or "unknown"
        goal_type = (row.get("goal_type") or "").strip() or "unknown"
        run_dir = runs_dir / f"{geo_id}__{question_id}"
        report_exists = (run_dir / "report.md").exists()

        if run_dir.exists():
            cited, mentioned = compute_metrics_for_run(run_dir)
        else:
            cited, mentioned = False, False
            missing_run_dirs += 1

        by_domain_total[domain] += 1
        by_geo_method_total[geo_method] += 1
        by_goal_type_total[goal_type] += 1

        if report_exists:
            generated_reports += 1
            by_domain_report[domain] += 1
        if cited:
            by_domain_cited[domain] += 1
            by_geo_method_cited[geo_method] += 1
            by_goal_type_cited[goal_type] += 1
        if mentioned:
            by_domain_mentioned[domain] += 1
            by_geo_method_mentioned[geo_method] += 1
            by_goal_type_mentioned[goal_type] += 1

        new_row = dict(row)
        new_row["cited"] = "true" if cited else "false"
        new_row["mentioned"] = "true" if mentioned else "false"
        output_rows.append(new_row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Dataset: {dataset_csv}")
    print(f"Runs: {runs_dir}")
    print(f"Wrote: {output_csv}")
    print(f"Rows: {len(output_rows)}")
    print(f"Missing run directories: {missing_run_dirs}")
    print(f"Generated report.md files: {generated_reports}/{len(output_rows)}")

    print("\nPer-domain summary:")
    for domain in sorted(by_domain_total):
        total = by_domain_total[domain]
        report_count = by_domain_report[domain]
        cited_count = by_domain_cited[domain]
        mentioned_count = by_domain_mentioned[domain]
        print(
            f"  {domain}: reports={report_count}/{total}, "
            f"cited={_rate(cited_count, total)}, mentioned={_rate(mentioned_count, total)}"
        )

    print("\nBy geo_method:")
    for method in sorted(by_geo_method_total):
        total = by_geo_method_total[method]
        print(
            f"  {method}: "
            f"cited={_rate(by_geo_method_cited[method], total)}, "
            f"mentioned={_rate(by_geo_method_mentioned[method], total)}"
        )

    print("\nBy goal_type:")
    for goal in sorted(by_goal_type_total):
        total = by_goal_type_total[goal]
        print(
            f"  {goal}: "
            f"cited={_rate(by_goal_type_cited[goal], total)}, "
            f"mentioned={_rate(by_goal_type_mentioned[goal], total)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
