"""
Demo script that injects a manual document into ONLY the first retrieval result.

This keeps the standard Co-STORM demo behavior while giving you a simple hook
to edit a JSON file and force one of the retrieved documents.
"""

from __future__ import annotations

import json
import os
import re
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from manual_document_helper import create_information_from_dict

from knowledge_storm.collaborative_storm.engine import (
    CollaborativeStormLMConfigs, CoStormRunner, RunnerArgument)
from knowledge_storm.collaborative_storm.modules.callback import \
    LocalConsolePrintCallBackHandler
from knowledge_storm.interface import Information
from knowledge_storm.lm import AzureOpenAIModel, OpenAIModel
from knowledge_storm.logging_wrapper import LoggingWrapper
from knowledge_storm.rm import (BingSearch, BraveRM, DuckDuckGoSearchRM,
                                GoogleSearch, LoggingRetriever, SearXNG,
                                SerperRM, TavilySearchRM, YouRM)
from knowledge_storm.utils import load_api_key


class FirstRetrievalInjector:
    """
    Wraps a retriever so that the provided manual documents are injected exactly once:
    on the first retrieval call. All later calls go straight to the base retriever.
    """

    def __init__(self, base_retriever, manual_documents: List[Information]):
        self.base_retriever = base_retriever
        self.manual_documents = manual_documents
        self.injected = False
        if hasattr(base_retriever, "k"):
            self.k = base_retriever.k

    def _manual_results(self):
        results = []
        for doc in self.manual_documents:
            results.append(
                {
                    "url": doc.url,
                    "title": doc.title,
                    "description": doc.description,
                    "snippets": doc.snippets,
                    "meta": doc.meta,
                }
            )
        return results

    def forward(self, query_or_queries, exclude_urls=None):
        results = self.base_retriever.forward(
            query_or_queries=query_or_queries, exclude_urls=exclude_urls
        )
        if not self.injected and self.manual_documents:
            self.injected = True
            manual_results = self._manual_results()
            print(
                f"✓ Injected {len(manual_results)} manual document(s) into the first retrieval call."
            )
            return manual_results + results
        return results

    __call__ = forward


def load_manual_documents(json_path: Path) -> List[Information]:
    if not json_path.exists():
        raise FileNotFoundError(f"Manual document file not found: {json_path}")
    with open(json_path, "r") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = [data]
    documents = [create_information_from_dict(item) for item in data]
    print(
        f"✓ Loaded {len(documents)} manual document(s) from {json_path.as_posix()}"
    )
    return documents


def extract_report_title(article: Optional[str]) -> Optional[str]:
    if not article:
        return None
    for line in article.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                return title
    return None


def slugify_for_path(value: Optional[str]) -> str:
    if not value:
        return "unnamed-report"
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value or "unnamed-report"


def configure_models():
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

    # Use cheaper/faster models (same defaults as the demo)
    gpt_4o_mini_model_name = "gpt-4o-mini"

    lm_config.set_question_answering_lm(
        ModelClass(model=gpt_4o_mini_model_name, max_tokens=1000, **openai_kwargs)
    )
    lm_config.set_discourse_manage_lm(
        ModelClass(model=gpt_4o_mini_model_name, max_tokens=500, **openai_kwargs)
    )
    lm_config.set_utterance_polishing_lm(
        ModelClass(model=gpt_4o_mini_model_name, max_tokens=2000, **openai_kwargs)
    )
    lm_config.set_warmstart_outline_gen_lm(
        ModelClass(model=gpt_4o_mini_model_name, max_tokens=500, **openai_kwargs)
    )
    lm_config.set_question_asking_lm(
        ModelClass(model=gpt_4o_mini_model_name, max_tokens=300, **openai_kwargs)
    )
    lm_config.set_knowledge_base_lm(
        ModelClass(model=gpt_4o_mini_model_name, max_tokens=1000, **openai_kwargs)
    )
    return lm_config


def build_retriever(args, runner_argument, manual_docs: List[Information], logging_wrapper=None):
    # Default retriever: Google Custom Search
    args.retriever = args.retriever or "google"
    match args.retriever:
        case "google":
            base_rm = GoogleSearch(
                google_search_api_key=os.getenv("GOOGLE_SEARCH_API_KEY"),
                google_cse_id=os.getenv("GOOGLE_CSE_ID"),
                k=runner_argument.retrieve_top_k,
            )
        case "bing":
            base_rm = BingSearch(
                bing_search_api=os.getenv("BING_SEARCH_API_KEY"),
                k=runner_argument.retrieve_top_k,
            )
        case "you":
            base_rm = YouRM(
                ydc_api_key=os.getenv("YDC_API_KEY"),
                k=runner_argument.retrieve_top_k,
            )
        case "brave":
            base_rm = BraveRM(
                brave_search_api_key=os.getenv("BRAVE_API_KEY"),
                k=runner_argument.retrieve_top_k,
            )
        case "duckduckgo":
            base_rm = DuckDuckGoSearchRM(
                k=runner_argument.retrieve_top_k, safe_search="On", region="us-en"
            )
        case "serper":
            base_rm = SerperRM(
                serper_search_api_key=os.getenv("SERPER_API_KEY"),
                query_params={"autocorrect": True, "num": 10, "page": 1},
            )
        case "tavily":
            base_rm = TavilySearchRM(
                tavily_search_api_key=os.getenv("TAVILY_API_KEY"),
                k=runner_argument.retrieve_top_k,
                include_raw_content=True,
            )
        case "searxng":
            base_rm = SearXNG(
                searxng_api_key=os.getenv("SEARXNG_API_KEY"),
                k=runner_argument.retrieve_top_k,
            )
        case _:
            raise ValueError(
                f'Invalid retriever: {args.retriever}. '
                'Choose from "bing", "you", "brave", "duckduckgo", '
                '"serper", "tavily", "searxng", or "google".'
            )

    injector = FirstRetrievalInjector(base_rm, manual_docs)
    if logging_wrapper is not None:
        return LoggingRetriever(injector, logging_wrapper)
    return injector


def main(args):
    lm_config = configure_models()

    topic = input("Topic: ").strip()
    runner_argument = RunnerArgument(
        topic=topic,
        retrieve_top_k=args.retrieve_top_k,
        max_search_queries=args.max_search_queries,
        total_conv_turn=args.total_conv_turn,
        max_search_thread=args.max_search_thread,
        max_search_queries_per_turn=args.max_search_queries_per_turn,
        warmstart_max_num_experts=args.warmstart_max_num_experts,
        warmstart_max_turn_per_experts=args.warmstart_max_turn_per_experts,
        warmstart_max_thread=args.warmstart_max_thread,
        max_thread_num=args.max_thread_num,
        max_num_round_table_experts=args.max_num_round_table_experts,
        moderator_override_N_consecutive_answering_turn=args.moderator_override_N_consecutive_answering_turn,
        node_expansion_trigger_count=args.node_expansion_trigger_count,
    )

    logging_wrapper = LoggingWrapper(lm_config)
    callback_handler = (
        LocalConsolePrintCallBackHandler() if args.enable_log_print else None
    )

    manual_doc_path = Path(args.manual_doc_path).expanduser().resolve()
    manual_docs = load_manual_documents(manual_doc_path)
    retriever = build_retriever(args, runner_argument, manual_docs, logging_wrapper)

    costorm_runner = CoStormRunner(
        lm_config=lm_config,
        runner_argument=runner_argument,
        logging_wrapper=logging_wrapper,
        rm=retriever,
        callback_handler=callback_handler,
    )

    costorm_runner.warm_start()

    print("\n" + "=" * 80)
    print("INTERACTIVE CONVERSATION")
    print("=" * 80 + "\n")
    for _ in range(args.demo_turns):
        conv_turn = costorm_runner.step()
        print(f"**{conv_turn.role}**: {conv_turn.utterance}\n")

    article = None
    if not args.skip_report:
        print("\n" + "=" * 80)
        print("Generating report...")
        print("=" * 80 + "\n")
        costorm_runner.knowledge_base.reorganize()
        article = costorm_runner.generate_report()
        if article:
            print(article[:2000] + ("..." if len(article) > 2000 else ""))

    report_title = extract_report_title(article)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_slug = slugify_for_path(report_title or topic or "no-topic")
    output_folder_name = f"{timestamp}_{folder_slug}"
    final_output_dir = os.path.join(args.output_dir, output_folder_name)
    os.makedirs(final_output_dir, exist_ok=True)
    print(
        f"\n✓ Output folder '{output_folder_name}' created at {timestamp} "
        f"(title: {report_title or 'N/A'})"
    )

    if article:
        with open(os.path.join(final_output_dir, "report.md"), "w") as f:
            f.write(article)
        print(f"✓ Report saved to {os.path.join(final_output_dir, 'report.md')}")

    with open(os.path.join(final_output_dir, "instance_dump.json"), "w") as f:
        json.dump(costorm_runner.to_dict(), f, indent=2)
    with open(os.path.join(final_output_dir, "log.json"), "w") as f:
        json.dump(costorm_runner.dump_logging_and_reset(), f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Co-STORM demo with a manual document injected into the first retrieval call."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/co-storm-demo",
        help="Directory to store timestamped outputs.",
    )
    parser.add_argument(
        "--retriever",
        type=str,
        choices=["bing", "you", "brave", "serper", "duckduckgo", "tavily", "searxng", "google"],
        help="Search API to use (defaults to google).",
    )
    parser.add_argument(
        "--manual-doc-path",
        type=str,
        default=(
            Path(__file__).with_name("manual_document_example.json").as_posix()
        ),
        help="Path to JSON file whose documents will be injected on the first retrieval call.",
    )
    parser.add_argument(
        "--retrieve_top_k",
        type=int,
        default=3,
        help="Retrieve top k results for each query.",
    )
    parser.add_argument(
        "--max_search_queries",
        type=int,
        default=1,
        help="Maximum number of search queries per question.",
    )
    parser.add_argument(
        "--total_conv_turn",
        type=int,
        default=5,
        help="Maximum number of turns in the conversation.",
    )
    parser.add_argument(
        "--max_search_thread",
        type=int,
        default=3,
        help="Maximum number of parallel threads for the retriever.",
    )
    parser.add_argument(
        "--max_search_queries_per_turn",
        type=int,
        default=2,
        help="Maximum number of search queries to consider per turn.",
    )
    parser.add_argument(
        "--warmstart_max_num_experts",
        type=int,
        default=1,
        help="Max number of experts during warm start.",
    )
    parser.add_argument(
        "--warmstart_max_turn_per_experts",
        type=int,
        default=1,
        help="Max number of turns per warm-start expert.",
    )
    parser.add_argument(
        "--warmstart_max_thread",
        type=int,
        default=1,
        help="Max threads for warm-start perspective QA.",
    )
    parser.add_argument(
        "--max_thread_num",
        type=int,
        default=5,
        help="Maximum number of threads used overall.",
    )
    parser.add_argument(
        "--max_num_round_table_experts",
        type=int,
        default=1,
        help="Max number of active experts in the round table.",
    )
    parser.add_argument(
        "--moderator_override_N_consecutive_answering_turn",
        type=int,
        default=2,
        help="Moderator override threshold for consecutive expert turns.",
    )
    parser.add_argument(
        "--node_expansion_trigger_count",
        type=int,
        default=10,
        help="Trigger node expansion for nodes exceeding this snippet count.",
    )
    parser.add_argument(
        "--demo-turns",
        type=int,
        default=2,
        help="Number of interactive turns to run after warm start.",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip final report generation to save costs.",
    )
    parser.add_argument(
        "--enable_log_print",
        action="store_true",
        help="Enable console logging callbacks.",
    )

    main(parser.parse_args())

