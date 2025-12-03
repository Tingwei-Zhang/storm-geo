"""
Demo version of Co-STORM with reduced costs for quick demonstrations.
This script uses minimal parameters to reduce API calls and costs.
"""

import os
import re
import json
from datetime import datetime
from argparse import ArgumentParser
from typing import Optional
from knowledge_storm.collaborative_storm.engine import (
    CollaborativeStormLMConfigs,
    RunnerArgument,
    CoStormRunner,
)
from knowledge_storm.collaborative_storm.modules.callback import (
    LocalConsolePrintCallBackHandler,
)
from knowledge_storm.lm import OpenAIModel, AzureOpenAIModel
from knowledge_storm.logging_wrapper import LoggingWrapper
from knowledge_storm.rm import (
    YouRM,
    BingSearch,
    BraveRM,
    SerperRM,
    DuckDuckGoSearchRM,
    TavilySearchRM,
    SearXNG,
)
from knowledge_storm.utils import load_api_key


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


def main(args):
    load_api_key(toml_file_path="secrets.toml")
    lm_config: CollaborativeStormLMConfigs = CollaborativeStormLMConfigs()
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

    ModelClass = (
        OpenAIModel if os.getenv("OPENAI_API_TYPE") == "openai" else AzureOpenAIModel
    )
    
    # Use cheaper/faster model for demo
    gpt_4o_mini_model_name = "gpt-4o-mini"
    gpt_4o_model_name = "gpt-4o"
    
    if os.getenv("OPENAI_API_TYPE") == "azure":
        openai_kwargs["api_base"] = os.getenv("AZURE_API_BASE")
        openai_kwargs["api_version"] = os.getenv("AZURE_API_VERSION")

    # Use cheaper models for most tasks (gpt-4o-mini is ~10x cheaper)
    # Only use stronger model for critical tasks
    question_answering_lm = ModelClass(
        model=gpt_4o_mini_model_name, max_tokens=1000, **openai_kwargs
    )
    discourse_manage_lm = ModelClass(
        model=gpt_4o_mini_model_name, max_tokens=500, **openai_kwargs
    )
    utterance_polishing_lm = ModelClass(
        model=gpt_4o_mini_model_name, max_tokens=2000, **openai_kwargs
    )
    warmstart_outline_gen_lm = ModelClass(
        model=gpt_4o_mini_model_name, max_tokens=500, **openai_kwargs
    )
    question_asking_lm = ModelClass(
        model=gpt_4o_mini_model_name, max_tokens=300, **openai_kwargs
    )
    knowledge_base_lm = ModelClass(
        model=gpt_4o_mini_model_name, max_tokens=1000, **openai_kwargs
    )

    lm_config.set_question_answering_lm(question_answering_lm)
    lm_config.set_discourse_manage_lm(discourse_manage_lm)
    lm_config.set_utterance_polishing_lm(utterance_polishing_lm)
    lm_config.set_warmstart_outline_gen_lm(warmstart_outline_gen_lm)
    lm_config.set_question_asking_lm(question_asking_lm)
    lm_config.set_knowledge_base_lm(knowledge_base_lm)

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

    args.retriever='serper'
    # Co-STORM is a knowledge curation system which consumes information from the retrieval module.
    match args.retriever:
        case "bing":
            rm = BingSearch(
                bing_search_api=os.getenv("BING_SEARCH_API_KEY"),
                k=runner_argument.retrieve_top_k,
            )
        case "you":
            rm = YouRM(
                ydc_api_key=os.getenv("YDC_API_KEY"), k=runner_argument.retrieve_top_k
            )
        case "brave":
            rm = BraveRM(
                brave_search_api_key=os.getenv("BRAVE_API_KEY"),
                k=runner_argument.retrieve_top_k,
            )
        case "duckduckgo":
            rm = DuckDuckGoSearchRM(
                k=runner_argument.retrieve_top_k, safe_search="On", region="us-en"
            )
        case "serper":
            rm = SerperRM(
                serper_search_api_key=os.getenv("SERPER_API_KEY"),
                query_params={"autocorrect": True, "num": 10, "page": 1},
            )
        case "tavily":
            rm = TavilySearchRM(
                tavily_search_api_key=os.getenv("TAVILY_API_KEY"),
                k=runner_argument.retrieve_top_k,
                include_raw_content=True,
            )
        case "searxng":
            rm = SearXNG(
                searxng_api_key=os.getenv("SEARXNG_API_KEY"),
                k=runner_argument.retrieve_top_k,
            )
        case _:
            raise ValueError(
                f'Invalid retriever: {args.retriever}. Choose either "bing", "you", "brave", "duckduckgo", "serper", "tavily", or "searxng"'
            )

    costorm_runner = CoStormRunner(
        lm_config=lm_config,
        runner_argument=runner_argument,
        logging_wrapper=logging_wrapper,
        rm=rm,
        callback_handler=callback_handler,
    )

    # warm start the system
    costorm_runner.warm_start()

    # Demo: just observe 1-2 turns
    for _ in range(args.demo_turns):
        conv_turn = costorm_runner.step()
        print(f"**{conv_turn.role}**: {conv_turn.utterance}\n")

    # Generate report (default behavior, can be skipped with --skip-report)
    article = None
    if not args.skip_report:
        print("\n" + "="*80)
        print("Generating report...")
        print("="*80 + "\n")
        costorm_runner.knowledge_base.reorganize()
        article = costorm_runner.generate_report()
        print("\n" + "="*80)
        print("GENERATED REPORT (first 2000 chars):")
        print("="*80)
        print(article[:2000] + "..." if len(article) > 2000 else article)

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

    # Save article if generated (default behavior)
    if article:
        with open(os.path.join(final_output_dir, "report.md"), "w") as f:
            f.write(article)
        print(f"✓ Report saved to {os.path.join(final_output_dir, 'report.md')}")

    # Save instance dump
    instance_copy = costorm_runner.to_dict()
    with open(os.path.join(final_output_dir, "instance_dump.json"), "w") as f:
        json.dump(instance_copy, f, indent=2)

    # Save logging
    log_dump = costorm_runner.dump_logging_and_reset()
    with open(os.path.join(final_output_dir, "log.json"), "w") as f:
        json.dump(log_dump, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Co-STORM demo with reduced costs for quick demonstrations"
    )
    # global arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/co-storm-demo",
        help="Directory to store the outputs.",
    )
    parser.add_argument(
        "--retriever",
        type=str,
        choices=["bing", "you", "brave", "serper", "duckduckgo", "tavily", "searxng"],
        help="The search engine API to use for retrieving information.",
    )
    
    # DEMO-OPTIMIZED: Reduced parameters for cost savings
    parser.add_argument(
        "--retrieve_top_k",
        type=int,
        default=3,  # Reduced from 10 to 3 (70% reduction in documents)
        help="Retrieve top k results for each query in retriever.",
    )
    parser.add_argument(
        "--max_search_queries",
        type=int,
        default=1,  # Reduced from 2 to 1 (50% reduction in queries)
        help="Maximum number of search queries to consider for each question.",
    )
    parser.add_argument(
        "--total_conv_turn",
        type=int,
        default=5,  # Reduced from 20 to 5 (75% reduction)
        help="Maximum number of turns in conversation.",
    )
    parser.add_argument(
        "--max_search_thread",
        type=int,
        default=3,  # Reduced from 5 to 3
        help="Maximum number of parallel threads for retriever.",
    )
    parser.add_argument(
        "--max_search_queries_per_turn",
        type=int,
        default=2,  # Reduced from 3 to 2
        help="Maximum number of search queries to consider in each turn.",
    )
    parser.add_argument(
        "--warmstart_max_num_experts",
        type=int,
        default=1,  # Reduced from 3 to 1 (67% reduction)
        help="Max number of experts in perspective-guided QA during warm start.",
    )
    parser.add_argument(
        "--warmstart_max_turn_per_experts",
        type=int,
        default=1,  # Reduced from 2 to 1 (50% reduction)
        help="Max number of turns per perspective during warm start.",
    )
    parser.add_argument(
        "--warmstart_max_thread",
        type=int,
        default=1,  # Reduced from 3 to 1
        help="Max number of threads for parallel perspective-guided QA during warm start.",
    )
    parser.add_argument(
        "--max_thread_num",
        type=int,
        default=5,  # Reduced from 10 to 5
        help=(
            "Maximum number of threads to use. "
            "Consider reducing it if you keep getting 'Exceed rate limit' errors when calling the LM API."
        ),
    )
    parser.add_argument(
        "--max_num_round_table_experts",
        type=int,
        default=1,  # Reduced from 2 to 1
        help="Max number of active experts in round table discussion.",
    )
    parser.add_argument(
        "--moderator_override_N_consecutive_answering_turn",
        type=int,
        default=2,  # Reduced from 3 to 2
        help=(
            "Number of consecutive expert answering turns before the moderator overrides the conversation."
        ),
    )
    parser.add_argument(
        "--node_expansion_trigger_count",
        type=int,
        default=10,
        help="Trigger node expansion for nodes that contain more than N snippets.",
    )
    
    # Demo-specific options
    parser.add_argument(
        "--demo-turns",
        type=int,
        default=2,
        help="Number of conversation turns to observe in demo mode.",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="If set, skip report generation to save costs (report is generated by default).",
    )

    # Boolean flags
    parser.add_argument(
        "--enable_log_print",
        action="store_true",
        help="If set, enable console log print.",
    )

    main(parser.parse_args())

