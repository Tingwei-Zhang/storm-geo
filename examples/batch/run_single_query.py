"""
Run a single query through Co-STORM (warm start + N steps + report).

Accepts --question-id + --topic or --manifest-row. Optional injection doc via
--injection-doc-path or --injection-doc-s3-uri (or from manifest row).
Outputs: output_dir / sanitized_question_id / {report.md, instance_dump.json, log.json}.
Skips if output already exists. Sanitizes question_id for filesystem safety.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional

# Import create_information_from_dict from manual_examples (same format as manual doc JSON).
# Run from project root with PYTHONPATH=. so examples.manual_examples is resolvable.
from examples.manual_examples.manual_document_helper import \
    create_information_from_dict
from knowledge_storm.collaborative_storm.engine import (
    CollaborativeStormLMConfigs, CoStormRunner, RunnerArgument)
from knowledge_storm.interface import Information
from knowledge_storm.lm import AzureOpenAIModel, OpenAIModel
from knowledge_storm.logging_wrapper import LoggingWrapper
from knowledge_storm.rm import (BingSearch, BraveRM, DuckDuckGoSearchRM,
                                GoogleSearch, LoggingRetriever, SearXNG,
                                SerperRM, TavilySearchRM, YouRM)
from knowledge_storm.utils import load_api_key

from ._injector import FirstRetrievalInjector


def sanitize_question_id(question_id: str) -> str:
    """Make question_id safe for use as a directory name (alphanumeric, hyphen, underscore)."""
    if not question_id:
        return "unknown"
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", question_id)
    safe = safe.strip("_") or "unknown"
    return safe[:200]  # cap length


def load_injection_doc_from_path(path: Path) -> List[Information]:
    """Load injection document(s) from a local JSON file. Same format as manual_document_example.json."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    return [create_information_from_dict(item) for item in data]


def load_injection_doc_from_s3(s3_uri: str) -> List[Information]:
    """Load injection document(s) from S3. Downloads to temp file then loads JSON."""
    try:
        import boto3
    except ImportError:
        raise RuntimeError("Loading from S3 requires boto3. Install with: pip install boto3")
    match = re.match(r"s3://([^/]+)/(.+)$", s3_uri.strip())
    if not match:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    bucket, key = match.group(1), match.group(2)
    client = boto3.client("s3")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        try:
            client.download_fileobj(bucket, key, f)
            f.flush()
            f.close()
            with open(f.name, "r", encoding="utf-8") as fp:
                data = json.load(fp)
        finally:
            os.unlink(f.name)
    if isinstance(data, dict):
        data = [data]
    return [create_information_from_dict(item) for item in data]


def load_injection_docs(
    injection_doc_path: Optional[str] = None,
    injection_doc_s3_uri: Optional[str] = None,
) -> List[Information]:
    """Load injection document(s) from local path or S3 URI. Returns empty list if neither set."""
    if injection_doc_s3_uri:
        return load_injection_doc_from_s3(injection_doc_s3_uri)
    if injection_doc_path:
        path = Path(injection_doc_path).expanduser().resolve()
        if path.exists():
            return load_injection_doc_from_path(path)
    return []


def configure_models(lm_preset: str = "demo") -> CollaborativeStormLMConfigs:
    """Configure LM (demo = gpt-4o-mini for all; gpt = gpt-4o for main components)."""
    load_api_key(toml_file_path="secrets.toml")
    lm_config = CollaborativeStormLMConfigs()
    openai_kwargs = (
        {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "api_provider": "openai",
            "temperature": 1.0,
            "top_p": 0.9,
            "api_base": None,
        }
        if os.getenv("OPENAI_API_TYPE") == "openai"
        else {
            "api_key": os.getenv("AZURE_API_KEY"),
            "temperature": 1.0,
            "top_p": 0.9,
            "api_base": os.getenv("AZURE_API_BASE"),
            "api_version": os.getenv("AZURE_API_VERSION"),
        }
    )
    if os.getenv("OPENAI_API_TYPE") == "azure":
        openai_kwargs["api_base"] = os.getenv("AZURE_API_BASE")
        openai_kwargs["api_version"] = os.getenv("AZURE_API_VERSION")
    ModelClass = (
        OpenAIModel if os.getenv("OPENAI_API_TYPE") == "openai" else AzureOpenAIModel
    )
    model_name = "gpt-4o-mini" if lm_preset == "demo" else "gpt-4o"
    lm_config.set_question_answering_lm(
        ModelClass(model=model_name, max_tokens=1000, **openai_kwargs)
    )
    lm_config.set_discourse_manage_lm(
        ModelClass(model=model_name, max_tokens=500, **openai_kwargs)
    )
    lm_config.set_utterance_polishing_lm(
        ModelClass(model=model_name, max_tokens=2000, **openai_kwargs)
    )
    lm_config.set_warmstart_outline_gen_lm(
        ModelClass(model=model_name, max_tokens=500, **openai_kwargs)
    )
    lm_config.set_question_asking_lm(
        ModelClass(model=model_name, max_tokens=300, **openai_kwargs)
    )
    lm_config.set_knowledge_base_lm(
        ModelClass(model=model_name, max_tokens=1000, **openai_kwargs)
    )
    return lm_config


def build_retriever(
    retriever_name: str,
    retrieve_top_k: int,
    manual_docs: List[Information],
    logging_wrapper: Optional[LoggingWrapper],
):
    """Build retriever: base_rm, optionally FirstRetrievalInjector, then LoggingRetriever."""
    retriever_name = retriever_name or "google"
    if retriever_name == "google":
        base_rm = GoogleSearch(
            google_search_api_key=os.getenv("GOOGLE_SEARCH_API_KEY"),
            google_cse_id=os.getenv("GOOGLE_CSE_ID"),
            k=retrieve_top_k,
        )
    elif retriever_name == "bing":
        base_rm = BingSearch(
            bing_search_api_key=os.getenv("BING_SEARCH_API_KEY"),
            k=retrieve_top_k,
        )
    elif retriever_name == "you":
        base_rm = YouRM(
            ydc_api_key=os.getenv("YDC_API_KEY"),
            k=retrieve_top_k,
        )
    elif retriever_name == "brave":
        base_rm = BraveRM(
            brave_search_api_key=os.getenv("BRAVE_API_KEY"),
            k=retrieve_top_k,
        )
    elif retriever_name == "duckduckgo":
        base_rm = DuckDuckGoSearchRM(
            k=retrieve_top_k, safe_search="On", region="us-en"
        )
    elif retriever_name == "serper":
        base_rm = SerperRM(
            serper_search_api_key=os.getenv("SERPER_API_KEY"),
            query_params={"autocorrect": True, "num": 10, "page": 1},
        )
    elif retriever_name == "tavily":
        base_rm = TavilySearchRM(
            tavily_search_api_key=os.getenv("TAVILY_API_KEY"),
            k=retrieve_top_k,
            include_raw_content=True,
        )
    elif retriever_name == "searxng":
        base_rm = SearXNG(
            searxng_api_key=os.getenv("SEARXNG_API_KEY"),
            k=retrieve_top_k,
        )
    else:
        raise ValueError(
            f"Invalid retriever: {retriever_name}. "
            'Choose from "bing", "you", "brave", "duckduckgo", "serper", "tavily", "searxng", "google".'
        )
    if manual_docs:
        base_rm = FirstRetrievalInjector(base_rm, manual_docs)
    if logging_wrapper is not None:
        return LoggingRetriever(base_rm, logging_wrapper)
    return base_rm


def run_single_query(
    question_id: str,
    topic: str,
    output_dir: Path,
    *,
    injection_doc_path: Optional[str] = None,
    injection_doc_s3_uri: Optional[str] = None,
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
    moderator_override_N_consecutive_answering_turn: int = 2,
    node_expansion_trigger_count: int = 10,
    lm_preset: str = "demo",
    skip_if_exists: bool = True,
) -> bool:
    """
    Run one query through Co-STORM. Returns True on success, False on failure.
    Writes report.md, instance_dump.json, log.json under output_dir / sanitized_question_id.
    """
    safe_id = sanitize_question_id(question_id)
    out_query_dir = output_dir / safe_id
    sentinel = out_query_dir / "report.md"
    if skip_if_exists and sentinel.exists():
        return True

    manual_docs = load_injection_docs(
        injection_doc_path=injection_doc_path,
        injection_doc_s3_uri=injection_doc_s3_uri,
    )
    lm_config = configure_models(lm_preset=lm_preset)
    logging_wrapper = LoggingWrapper(lm_config)
    runner_argument = RunnerArgument(
        topic=topic,
        retrieve_top_k=retrieve_top_k,
        max_search_queries=max_search_queries,
        total_conv_turn=total_conv_turn,
        max_search_thread=max_search_thread,
        max_search_queries_per_turn=max_search_queries_per_turn,
        warmstart_max_num_experts=warmstart_max_num_experts,
        warmstart_max_turn_per_experts=warmstart_max_turn_per_experts,
        warmstart_max_thread=warmstart_max_thread,
        max_thread_num=max_thread_num,
        max_num_round_table_experts=max_num_round_table_experts,
        moderator_override_N_consecutive_answering_turn=moderator_override_N_consecutive_answering_turn,
        node_expansion_trigger_count=node_expansion_trigger_count,
    )
    rm = build_retriever(
        retriever_name=retriever,
        retrieve_top_k=runner_argument.retrieve_top_k,
        manual_docs=manual_docs,
        logging_wrapper=logging_wrapper,
    )
    costorm_runner = CoStormRunner(
        lm_config=lm_config,
        runner_argument=runner_argument,
        logging_wrapper=logging_wrapper,
        rm=rm,
        callback_handler=None,
    )
    costorm_runner.warm_start()
    for _ in range(demo_turns):
        costorm_runner.step()
    costorm_runner.knowledge_base.reorganize()
    article = costorm_runner.generate_report()

    out_query_dir.mkdir(parents=True, exist_ok=True)
    if article:
        (out_query_dir / "report.md").write_text(article, encoding="utf-8")
    (out_query_dir / "instance_dump.json").write_text(
        json.dumps(costorm_runner.to_dict(), indent=2), encoding="utf-8"
    )
    (out_query_dir / "log.json").write_text(
        json.dumps(costorm_runner.dump_logging_and_reset(), indent=2), encoding="utf-8"
    )
    return True


def parse_manifest_row(manifest_row_path: Path) -> dict:
    """Parse single-row CSV or JSON into dict (question_id, topic, etc.)."""
    path = Path(manifest_row_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Manifest row file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, list) and data:
            return data[0]
        return data if isinstance(data, dict) else {}
    import csv
    from io import StringIO
    reader = csv.DictReader(StringIO(text))
    rows = list(reader)
    if not rows:
        raise ValueError(f"No row in manifest file: {path}")
    return {k.strip(): v for k, v in rows[0].items()}


def main() -> int:
    parser = ArgumentParser(
        description="Run a single query through Co-STORM (batch worker)."
    )
    parser.add_argument("--question-id", type=str, default=None)
    parser.add_argument("--topic", type=str, default=None)
    parser.add_argument(
        "--manifest-row",
        type=Path,
        default=None,
        help="Path to single-row CSV or JSON (question_id, topic, injection_doc_path/s3_uri).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--injection-doc-path", type=str, default=None)
    parser.add_argument("--injection-doc-s3-uri", type=str, default=None)
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
    parser.add_argument(
        "--moderator-override-n-consecutive-answering-turn", type=int, default=2
    )
    parser.add_argument("--node-expansion-trigger-count", type=int, default=10)
    parser.add_argument("--lm-preset", type=str, choices=["demo", "gpt"], default="demo")
    parser.add_argument("--no-skip-existing", action="store_true", help="Run even if output exists.")
    args = parser.parse_args()

    if args.manifest_row is not None:
        row = parse_manifest_row(args.manifest_row)
        question_id = row.get("question_id") or args.question_id
        topic = row.get("topic") or args.topic
        injection_doc_path = row.get("injection_doc_path") or args.injection_doc_path
        injection_doc_s3_uri = row.get("injection_doc_s3_uri") or args.injection_doc_s3_uri
    else:
        question_id = args.question_id
        topic = args.topic
        injection_doc_path = args.injection_doc_path
        injection_doc_s3_uri = args.injection_doc_s3_uri

    if not question_id or not topic:
        print(
            "Provide --question-id and --topic, or --manifest-row.",
            file=sys.stderr,
        )
        return 1

    try:
        ok = run_single_query(
            question_id=question_id,
            topic=topic,
            output_dir=args.output_dir,
            injection_doc_path=injection_doc_path,
            injection_doc_s3_uri=injection_doc_s3_uri,
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
            moderator_override_N_consecutive_answering_turn=(
                args.moderator_override_n_consecutive_answering_turn
            ),
            node_expansion_trigger_count=args.node_expansion_trigger_count,
            lm_preset=args.lm_preset,
            skip_if_exists=not args.no_skip_existing,
        )
        return 0 if ok else 1
    except Exception as e:
        print(f"run_single_query failed: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
