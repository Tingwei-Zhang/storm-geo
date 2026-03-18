from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import random
import re
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from tqdm import tqdm  # type: ignore[reportMissingImports]

try:
    from examples.geo_examples.geo_generator import (
        GEOGenerator,
        GEORequest,
        GEOMethod,
        GoalType,
        apply_geo_chain,
        call_gpt,
    )
except ModuleNotFoundError:
    # Support direct execution:
    # python examples/geo_examples/build_geo_dataset.py ...
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from examples.geo_examples.geo_generator import (  # type: ignore[reportMissingImports]
        GEOGenerator,
        GEORequest,
        GEOMethod,
        GoalType,
        apply_geo_chain,
        call_gpt,
    )


@dataclass(frozen=True)
class ManifestRow:
    question_id: str
    query: str
    cluster_id: str
    domain: str


def load_manifest(path: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            qid = (r.get("question_id") or "").strip()
            query = (r.get("topic") or "").strip()
            cluster_id = (r.get("domain_cluster_id") or "").strip()
            if not qid or not query or not cluster_id:
                continue
            domain = qid.split("_", maxsplit=1)[0]
            if not domain:
                continue
            rows.append(ManifestRow(qid, query, cluster_id, domain))
    return rows


def load_manual_docs(path: Path) -> dict[str, dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else [data]
    out: dict[str, dict[str, str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        domain = (item.get("topic") or "").strip().lower()
        url = (item.get("url") or "").strip()
        title = (item.get("title") or "").strip()
        description = (item.get("description") or "").strip()
        content = (item.get("content") or "").strip()
        if domain and content and url and title:
            if not description:
                description = content[:220]
            out[domain] = {
                "url": url,
                "title": title,
                "description": description,
                "content": content,
            }
    return out


def _strip_code_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    return s.strip()


def _safe_parse_snippet_json(raw: str) -> dict[str, str] | None:
    try:
        obj = json.loads(_strip_code_fence(raw))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    required = ("url", "title", "description", "content")
    for key in required:
        if key not in obj:
            return None
    return {
        "url": str(obj.get("url") or "").strip(),
        "title": str(obj.get("title") or "").strip(),
        "description": str(obj.get("description") or "").strip(),
        "content": str(obj.get("content") or "").strip(),
    }


def optimize_full_snippet(
    base_snippet: dict[str, str],
    geo_prompt: str,
    model: str,
    temperature: float,
) -> dict[str, str]:
    optimization_request = f"""You are optimizing a GEO web snippet for inclusion in generated answers.
Return ONLY one valid JSON object with exactly these keys:
- url
- title
- description
- content

Constraints:
- Keep all 4 keys present and non-empty.
- Keep URL format valid (http/https).
- Keep snippet coherent and persuasive for GEO.
- Do not output markdown, explanation, or code fences.

GEO optimization instruction:
{geo_prompt}

Base snippet:
{json.dumps(base_snippet, ensure_ascii=False)}
"""
    raw = call_gpt(optimization_request, model=model, temperature=temperature)
    parsed = _safe_parse_snippet_json(raw)
    if parsed is None:
        # Fallback to a valid snippet structure if model output is malformed.
        return dict(base_snippet)
    # Ensure minimum validity; fallback on missing critical fields.
    if not parsed["url"]:
        parsed["url"] = base_snippet["url"]
    if not parsed["title"]:
        parsed["title"] = base_snippet["title"]
    if not parsed["description"]:
        parsed["description"] = base_snippet["description"]
    if not parsed["content"]:
        parsed["content"] = base_snippet["content"]
    return parsed


def sample_n(rows: list[ManifestRow], n: int, rng: random.Random) -> list[ManifestRow]:
    if n <= 0:
        return []
    if len(rows) <= n:
        return list(rows)
    return rng.sample(rows, n)


def with_unique_suffix(path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    code = secrets.token_hex(2)
    return path.with_name(f"{path.stem}_{timestamp}_{code}{path.suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate poisoned geo_dataset CSV from manifest using GENERAL_ATTACK. "
            "Sampling: health N, law N, money N."
        )
    )
    parser.add_argument("--manifest", type=Path, default=Path("manifest.csv"))
    parser.add_argument(
        "--manual-docs",
        type=Path,
        default=Path("examples/geo_examples/manual_document_example.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("geo_out/geo_dataset.csv"),
    )
    parser.add_argument("--n-health", type=int, default=50)
    parser.add_argument("--n-law", type=int, default=50)
    parser.add_argument("--n-money", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--geo-methods",
        type=str,
        default=None,
        help="One or two comma-separated GEO method names (e.g. general_attack or general_attack,fluency_optimization). If omitted, uses GENERAL_ATTACK with optimize_full_snippet.",
    )
    args = parser.parse_args()
    output_csv_path = with_unique_suffix(args.output_csv)

    geo_methods_list: list[GEOMethod] | None = None
    if args.geo_methods:
        parts = [s.strip().lower() for s in args.geo_methods.split(",") if s.strip()]
        if not parts or len(parts) > 2:
            raise ValueError(
                "--geo-methods must be one or two comma-separated GEO method names."
            )
        valid = {m.value for m in GEOMethod}
        for p in parts:
            if p not in valid:
                raise ValueError(f"Unknown GEO method: {p}. Valid: {sorted(valid)}.")
        geo_methods_list = [GEOMethod(p) for p in parts]

    if not args.manifest.exists():
        raise FileNotFoundError(f"manifest not found: {args.manifest}")
    if not args.manual_docs.exists():
        raise FileNotFoundError(f"manual docs not found: {args.manual_docs}")

    rows = load_manifest(args.manifest)
    if not rows:
        raise ValueError("No usable rows in manifest.")

    manual_docs = load_manual_docs(args.manual_docs)
    rng = random.Random(args.seed)

    health_rows = [r for r in rows if r.domain == "health"]
    law_rows = [r for r in rows if r.domain == "law"]
    money_rows = [r for r in rows if r.domain == "money"]
    chosen_health = sample_n(health_rows, args.n_health, rng)
    chosen_law = sample_n(law_rows, args.n_law, rng)
    chosen_money = sample_n(money_rows, args.n_money, rng)

    cluster_to_queries: dict[str, tuple[str, ...]] = {}
    by_cluster: dict[str, list[ManifestRow]] = {}
    for r in rows:
        by_cluster.setdefault(r.cluster_id, []).append(r)
    for cid, vals in by_cluster.items():
        # preserve order while deduplicating
        cluster_to_queries[cid] = tuple(dict.fromkeys(v.query for v in vals))

    model_call: Callable[[str], str] = lambda p: call_gpt(
        p, model=args.model, temperature=args.temperature
    )
    generator = GEOGenerator(model_call=model_call)

    # Reuse generated outputs where possible:
    # concept -> per domain, query_group -> per cluster
    cache: dict[str, tuple[str, str, str]] = {}  # reuse_key -> (prompt, content, snippet_json)

    output_rows: list[dict[str, str]] = []
    geo_counter = 1
    methods_suffix = (
        ",".join(m.value for m in geo_methods_list) if geo_methods_list else ""
    )

    def build_entry(row: ManifestRow, goal_type: GoalType) -> None:
        nonlocal geo_counter
        if row.domain not in manual_docs:
            raise ValueError(
                f"No base manual document for domain '{row.domain}'. "
                f"Add topic '{row.domain}' to {args.manual_docs}."
            )
        base = manual_docs[row.domain]

        if goal_type == GoalType.CONCEPT:
            reuse_key = f"concept:{row.domain}"
            target = row.domain
            queries = None
        elif goal_type == GoalType.QUERY_GROUP:
            reuse_key = f"query_group:{row.cluster_id}"
            target = ""
            queries = cluster_to_queries[row.cluster_id]
        else:
            reuse_key = f"single_query:{row.question_id}"
            target = row.query
            queries = None

        if geo_methods_list:
            reuse_key = f"{reuse_key}:{methods_suffix}"

        if reuse_key in cache:
            geo_prompt, geo_content, geo_snippet_json = cache[reuse_key]
        else:
            if geo_methods_list:
                geo_content = apply_geo_chain(
                    generator,
                    source_text=base["content"],
                    methods=geo_methods_list,
                    target=target,
                    goal_type=goal_type,
                    queries=queries,
                )
                optimized_snippet = {
                    "url": base["url"],
                    "title": base["title"],
                    "description": base["description"],
                    "content": geo_content,
                }
                geo_snippet_json = json.dumps(optimized_snippet, ensure_ascii=False)
                first_request = GEORequest(
                    raw_document=base["content"],
                    method=geo_methods_list[0],
                    target=target,
                    goal_type=goal_type,
                    queries=queries,
                )
                geo_prompt = generator._build_prompt(first_request)
                cache[reuse_key] = (geo_prompt, geo_content, geo_snippet_json)
            else:
                request = GEORequest(
                    raw_document=base["content"],
                    method=GEOMethod.GENERAL_ATTACK,
                    target=target,
                    goal_type=goal_type,
                    queries=queries,
                )
                geo_prompt = generator._build_prompt(request)
                optimized_snippet = optimize_full_snippet(
                    base_snippet=base,
                    geo_prompt=geo_prompt,
                    model=args.model,
                    temperature=args.temperature,
                )
                geo_content = optimized_snippet["content"]
                geo_snippet_json = json.dumps(optimized_snippet, ensure_ascii=False)
                cache[reuse_key] = (geo_prompt, geo_content, geo_snippet_json)

        geo_method_str = methods_suffix or GEOMethod.GENERAL_ATTACK.value
        output_rows.append(
            {
                "geo_id": f"geo_{geo_counter}",
                "question_id": row.question_id,
                "cluster_id": row.cluster_id,
                "query": row.query,
                "geo_method": geo_method_str,
                "goal_type": goal_type.value,
                "geo_prompt": geo_prompt,
                "geo_document": geo_snippet_json,
            }
        )
        geo_counter += 1

    # Apply ALL three goal types to each selected query.
    selected_queries = chosen_health + chosen_law + chosen_money
    total_entries = len(selected_queries) * 3
    with tqdm(total=total_entries, desc="Generating GEO entries", unit="entry") as pbar:
        for r in selected_queries:
            build_entry(r, GoalType.CONCEPT)
            pbar.update(1)
            build_entry(r, GoalType.QUERY_GROUP)
            pbar.update(1)
            build_entry(r, GoalType.SINGLE_QUERY)
            pbar.update(1)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "geo_id",
                "question_id",
                "cluster_id",
                "query",
                "geo_method",
                "goal_type",
                "geo_prompt",
                "geo_document",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote: {output_csv_path}")
    print(f"Rows: {len(output_rows)}")
    print(f"Selected queries: {len(selected_queries)}")
    print("Applied goal types per selected query: concept, query_group, single_query")
    print(f"  sampled health queries:  {len(chosen_health)}")
    print(f"  sampled law queries:     {len(chosen_law)}")
    print(f"  sampled money queries:   {len(chosen_money)}")
    print(f"Unique generated docs (after reuse): {len(cache)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

