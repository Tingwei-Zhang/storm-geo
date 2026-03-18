"""
Build UGC injection dataset: manifest + per-cluster injection JSONs.
Reads manifest_test-style CSV, recurring_urls_raw.csv, domain descriptions, and products;
generates UGC-style content per URL and writes injection_{domain}_{cluster_id}.json + manifest.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Callable


def load_openai_api_key_from_secrets():
    # This code assumes secrets.toml is in the same directory as this script.
    secrets_path = Path(__file__).parent / "secrets.toml"
    if not secrets_path.exists():
        # Try repo root as fallback
        try:
            repo_root = Path(__file__).resolve().parents[2]
            secrets_path = repo_root / "secrets.toml"
        except Exception:
            pass
    if secrets_path.exists():
        import toml
        secrets = toml.load(secrets_path)
        key = secrets.get("OPENAI_API_KEY") or secrets.get("openai_api_key")
        if not key:
            # Try [openai] section
            section = secrets.get("openai") or {}
            key = section.get("api_key") or section.get("OPENAI_API_KEY")
        if key:
            os.environ["OPENAI_API_KEY"] = key


load_openai_api_key_from_secrets()

try:
    from examples.geo_examples.geo_generator import GEOMethod, call_gpt
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from examples.geo_examples.geo_generator import (  # type: ignore[no-redef]
        GEOMethod,
        call_gpt,
    )

from examples.ugc_injections.config import get_ugc_url_bases, url_base_to_category
from examples.ugc_injections.ugc_content_generator import (
    UGCContentGenerator,
    content_to_snippet_doc,
)


def parse_geo_methods(value: str) -> list[GEOMethod]:
    """Parse comma-separated method names into 1 or 2 GEOMethod values."""
    names = [s.strip().lower() for s in value.split(",") if s.strip()]
    if not names or len(names) > 2:
        raise ValueError(
            "--geo-methods must be one or two comma-separated GEO method names."
        )
    valid = {m.value for m in GEOMethod}
    out = []
    for n in names:
        if n not in valid:
            raise ValueError(
                f"Unknown GEO method: {n}. Valid: {sorted(valid)}."
            )
        out.append(GEOMethod(n))
    return out


def load_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): v for k, v in row.items()})
    return rows


def load_recurring_urls(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): v for k, v in row.items()})
    return rows


def filter_ugc_and_group(recurring_rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Group recurring rows by (dataset, cluster_id), keeping only UGC url_bases."""
    ugc_bases = get_ugc_url_bases()
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in recurring_rows:
        url_base = (row.get("url_base") or "").strip()
        if url_base not in ugc_bases:
            continue
        dataset = (row.get("dataset") or "").strip()
        cluster_id = (row.get("cluster_id") or "").strip()
        url = (row.get("url") or "").strip()
        if not dataset or not cluster_id or not url:
            continue
        key = (dataset, cluster_id)
        grouped.setdefault(key, []).append({"url": url, "url_base": url_base})
    return grouped


def load_descriptions(descriptions_dir: Path) -> dict[str, dict[str, str]]:
    """Load {domain}_descriptions.json; return domain -> {cluster_id: description}."""
    out = {}
    for path in descriptions_dir.glob("*_descriptions.json"):
        domain = path.stem.replace("_descriptions", "")
        data = json.loads(path.read_text(encoding="utf-8"))
        descs = data.get("descriptions") or {}
        out[domain] = {str(k): str(v) for k, v in descs.items()}
    return out


def load_products(products_path: Path) -> dict[str, str]:
    """Load manual_document_example.json; return topic (domain) -> content."""
    data = json.loads(products_path.read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else [data]
    out = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        topic = (item.get("topic") or "").strip().lower()
        content = (item.get("content") or "").strip()
        if topic and content:
            out[topic] = content
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build UGC injection dataset (manifest + injection JSONs per cluster)."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifest_test.csv"),
        help="Source manifest CSV (question_id, topic, domain, cluster_id).",
    )
    parser.add_argument(
        "--recurring-urls",
        type=Path,
        default=Path("recurring_urls_raw.csv"),
        help="Recurring URLs CSV with dataset, cluster_id, url, url_base.",
    )
    parser.add_argument(
        "--descriptions-dir",
        type=Path,
        default=Path("/Users/haltriedman/code/seo-geo/clustering_results/descriptions"),
        help="Directory containing {domain}_descriptions.json.",
    )
    parser.add_argument(
        "--products",
        type=Path,
        default=Path("examples/geo_examples/manual_document_example.json"),
        help="Products JSON (manual_document_example.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ugc_out"),
        help="Output directory for manifest and injection JSONs.",
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--ugc-mode",
        type=str,
        choices=("classic", "geo_only", "content_style_plus_geo"),
        default="classic",
        help="classic: forum/encyclopedic/qa per url_base; "
        "geo_only: source + 1 or 2 GEO methods; "
        "content_style_plus_geo: UGC style then 1 or 2 GEO methods.",
    )
    parser.add_argument(
        "--geo-methods",
        type=str,
        default=None,
        help="One or two comma-separated GEO method names (e.g. general_attack or general_attack,fluency_optimization). Required for geo_only and content_style_plus_geo.",
    )
    parser.add_argument(
        "--content-style",
        type=str,
        choices=("forum", "encyclopedic", "qa"),
        default=None,
        help="Required when --ugc-mode is content_style_plus_geo.",
    )
    args = parser.parse_args()

    if args.ugc_mode in ("geo_only", "content_style_plus_geo") and not args.geo_methods:
        print("--geo-methods required for --ugc-mode geo_only and content_style_plus_geo.", file=sys.stderr)
        return 1
    if args.ugc_mode == "content_style_plus_geo" and not args.content_style:
        print("--content-style required for --ugc-mode content_style_plus_geo.", file=sys.stderr)
        return 1
    geo_methods_list: list[GEOMethod] | None = None
    if args.geo_methods:
        try:
            geo_methods_list = parse_geo_methods(args.geo_methods)
        except ValueError as e:
            print(e, file=sys.stderr)
            return 1

    if not args.manifest.exists():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        return 1
    if not args.recurring_urls.exists():
        print(f"Recurring URLs not found: {args.recurring_urls}", file=sys.stderr)
        return 1
    if not args.descriptions_dir.exists():
        print(f"Descriptions dir not found: {args.descriptions_dir}", file=sys.stderr)
        return 1
    if not args.products.exists():
        print(f"Products not found: {args.products}", file=sys.stderr)
        return 1

    manifest_rows = load_manifest(args.manifest)
    recurring_rows = load_recurring_urls(args.recurring_urls)
    ugc_by_cluster = filter_ugc_and_group(recurring_rows)
    descriptions_by_domain = load_descriptions(args.descriptions_dir)
    products_by_domain = load_products(args.products)

    model_call: Callable[[str], str] = lambda p: call_gpt(
        user_prompt=p,
        model=args.model,
        temperature=args.temperature,
    )
    generator = UGCContentGenerator(model_call=model_call)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cluster_to_injection_path: dict[tuple[str, str], str] = {}

    for (domain, cluster_id), url_entries in ugc_by_cluster.items():
        cluster_desc = (descriptions_by_domain.get(domain) or {}).get(cluster_id)
        product_content = products_by_domain.get(domain)
        if not cluster_desc or not product_content:
            continue

        replacement_map = {}
        for entry in url_entries:
            url = entry["url"]
            url_base = entry["url_base"]
            try:
                if args.ugc_mode == "classic":
                    category = url_base_to_category(url_base)
                    if not category:
                        continue
                    content = generator.generate(
                        site_category=category,
                        cluster_description=cluster_desc,
                        product_content=product_content,
                    )
                elif args.ugc_mode == "geo_only":
                    content = generator.generate_geo_only(
                        product_content=product_content,
                        cluster_description=cluster_desc,
                        geo_methods=geo_methods_list,
                    )
                else:
                    assert args.ugc_mode == "content_style_plus_geo"
                    content = generator.generate_with_geo(
                        site_category=args.content_style,
                        cluster_description=cluster_desc,
                        product_content=product_content,
                        geo_methods=geo_methods_list,
                    )
            except Exception as e:
                print(
                    f"  Skip {url[:50]}...: {e}",
                    file=sys.stderr,
                )
                continue
            doc = content_to_snippet_doc(url=url, content=content)
            replacement_map[url] = doc

        if not replacement_map:
            continue
        injection_name = f"injection_{domain}_{cluster_id}.json"
        injection_path = args.output_dir / injection_name
        injection_path.write_text(
            json.dumps(replacement_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        cluster_to_injection_path[(domain, cluster_id)] = injection_name

    # Write manifest: copy original columns + ugc_injection_path
    manifest_out = args.output_dir / "manifest.csv"
    if not manifest_rows:
        print("No manifest rows.", file=sys.stderr)
        return 0

    fieldnames = list(manifest_rows[0].keys())
    if "ugc_injection_path" not in fieldnames:
        fieldnames.append("ugc_injection_path")

    with manifest_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in manifest_rows:
            domain = (row.get("domain") or "").strip()
            cluster_id = (row.get("cluster_id") or "").strip()
            ugc_path = cluster_to_injection_path.get((domain, cluster_id), "")
            row["ugc_injection_path"] = ugc_path
            writer.writerow(row)

    print(f"Wrote manifest: {manifest_out}")
    print(f"Injection files: {len(cluster_to_injection_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
