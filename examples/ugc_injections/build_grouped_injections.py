"""
Generate injection JSONs for the _grouped pipeline from grouped_ugc_clusters.csv.

For each selected group name, collects unique ugc_urls, generates UGC or GEO
content per URL (using group_name as cluster description and products JSON for
source content), and writes injection_{group}.json for use with run_local_parallel.

Uses examples.geo and UGCContentGenerator; supports --ugc-mode and --geo-methods
like build_ugc_dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Callable


def _load_secrets() -> None:
    for base in [Path(__file__).resolve().parent, Path(__file__).resolve().parents[2]]:
        p = base / "secrets.toml"
        if p.exists():
            try:
                import toml
                data = toml.load(p)
                key = data.get("OPENAI_API_KEY") or data.get("openai_api_key")
                if not key and "openai" in data:
                    key = (data["openai"] or {}).get("api_key") or (data["openai"] or {}).get("OPENAI_API_KEY")
                if key:
                    os.environ["OPENAI_API_KEY"] = str(key)
                    return
            except Exception:
                pass


_load_secrets()

try:
    from examples.geo_examples.geo_generator import GEOMethod, call_gpt
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from examples.geo_examples.geo_generator import GEOMethod, call_gpt  # type: ignore[no-redef]

from examples.ugc_injections.config import url_base_to_category
from examples.ugc_injections.ugc_content_generator import (
    UGCContentGenerator,
    content_to_snippet_doc,
)


def _sanitize_group_name(name: str) -> str:
    s = (name or "").strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[-\s]+", "_", s).strip("_")
    return s.lower() or "group"


def _parse_geo_methods(value: str) -> list[GEOMethod]:
    names = [s.strip().lower() for s in value.split(",") if s.strip()]
    if not names or len(names) > 2:
        raise ValueError("--geo-methods must be one or two comma-separated GEO method names.")
    valid = {m.value for m in GEOMethod}
    out = []
    for n in names:
        if n not in valid:
            raise ValueError(f"Unknown GEO method: {n}. Valid: {sorted(valid)}.")
        out.append(GEOMethod(n))
    return out


def _load_grouped_csv(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): v for k, v in row.items()})
    return rows


def _load_product_content(products_path: Path, topic: str | None) -> str:
    """Load one product content from manual_document_example.json. topic optional."""
    data = json.loads(products_path.read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else [data]
    for item in items:
        if not isinstance(item, dict):
            continue
        t = (item.get("topic") or "").strip().lower()
        content = (item.get("content") or "").strip()
        if not content:
            continue
        if topic and t != topic:
            continue
        return content
    return (items[0].get("content") or "").strip() if items and isinstance(items[0], dict) else ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate injection JSONs from grouped_ugc_clusters.csv for selected groups."
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
        help="Group names to include.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory (e.g. experiment_config/classic_grouped).",
    )
    parser.add_argument(
        "--products",
        type=Path,
        default=Path("examples/geo_examples/manual_document_example.json"),
        help="Products JSON for source content.",
    )
    parser.add_argument("--product-topic", type=str, default=None, help="Topic key for product (e.g. money, law, health).")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--ugc-mode",
        type=str,
        choices=("classic", "geo_only", "content_style_plus_geo"),
        default="classic",
    )
    parser.add_argument("--geo-methods", type=str, default=None)
    parser.add_argument("--content-style", type=str, choices=("forum", "encyclopedic", "qa"), default=None)
    args = parser.parse_args()

    if args.ugc_mode in ("geo_only", "content_style_plus_geo") and not args.geo_methods:
        print("--geo-methods required for geo_only and content_style_plus_geo.", file=sys.stderr)
        return 1
    if args.ugc_mode == "content_style_plus_geo" and not args.content_style:
        print("--content-style required for content_style_plus_geo.", file=sys.stderr)
        return 1
    geo_methods_list = None
    if args.geo_methods:
        try:
            geo_methods_list = _parse_geo_methods(args.geo_methods)
        except ValueError as e:
            print(e, file=sys.stderr)
            return 1

    if not args.grouped_csv.exists():
        print(f"Grouped CSV not found: {args.grouped_csv}", file=sys.stderr)
        return 1
    if not args.products.exists():
        print(f"Products not found: {args.products}", file=sys.stderr)
        return 1

    wanted = {g.strip() for g in args.group_names if g.strip()}
    rows = _load_grouped_csv(args.grouped_csv)
    product_content = _load_product_content(args.products, args.product_topic)
    if not product_content:
        print("No product content found.", file=sys.stderr)
        return 1

    model_call: Callable[[str], str] = lambda p: call_gpt(
        user_prompt=p, model=args.model, temperature=args.temperature
    )
    generator = UGCContentGenerator(model_call=model_call)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Per group_name: unique (ugc_url, ugc_domain) -> generate one doc per url
    by_group: dict[str, list[tuple[str, str]]] = {}
    for r in rows:
        gn = (r.get("group_name") or "").strip()
        if gn not in wanted:
            continue
        url = (r.get("ugc_url") or "").strip()
        domain = (r.get("ugc_domain") or "").strip().lower()
        if not url:
            continue
        by_group.setdefault(gn, [])
        seen = {(u, d) for u, d in by_group[gn]}
        if (url, domain) not in seen:
            by_group[gn].append((url, domain))

    for group_name, url_domain_list in by_group.items():
        safe = _sanitize_group_name(group_name)
        replacement_map = {}
        for url, ugc_domain in url_domain_list:
            category = url_base_to_category(ugc_domain)
            if args.ugc_mode == "classic":
                if not category:
                    continue
                try:
                    content = generator.generate(
                        site_category=category,
                        cluster_description=group_name,
                        product_content=product_content,
                    )
                except Exception as e:
                    print(f"  Skip {url[:60]}...: {e}", file=sys.stderr)
                    continue
            elif args.ugc_mode == "geo_only":
                try:
                    content = generator.generate_geo_only(
                        product_content=product_content,
                        cluster_description=group_name,
                        geo_methods=geo_methods_list,
                    )
                except Exception as e:
                    print(f"  Skip {url[:60]}...: {e}", file=sys.stderr)
                    continue
            else:
                if not category:
                    category = "forum"
                try:
                    content = generator.generate_with_geo(
                        site_category=category,
                        cluster_description=group_name,
                        product_content=product_content,
                        geo_methods=geo_methods_list,
                    )
                except Exception as e:
                    print(f"  Skip {url[:60]}...: {e}", file=sys.stderr)
                    continue
            doc = content_to_snippet_doc(url=url, content=content)
            replacement_map[url] = doc

        out_path = args.output_dir / f"injection_{safe}.json"
        out_path.write_text(json.dumps(replacement_map, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {out_path} ({len(replacement_map)} URLs)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
