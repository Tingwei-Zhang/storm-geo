#!/usr/bin/env python3
"""Generate a readable Markdown/HTML summary for a Co-STORM run directory."""
import argparse
import html
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_kv_table(data: Dict[str, object]) -> str:
    if not data:
        return "_No configuration data found._"
    lines = ["| Key | Value |", "| --- | --- |"]
    for key, value in data.items():
        lines.append(f"| {key} | {value} |")
    return "\n" + "\n".join(lines) + "\n"


def format_documents(docs: List[dict], doc_labels: Optional[Dict[str, str]] = None) -> str:
    if not docs:
        return "_None_"
    items = []
    for info in docs:
        url = info.get("url", "")
        title = info.get("title") or info.get("description", "Untitled")
        snippet = info.get("snippets", [""])[0] if info.get("snippets") else ""
        snippet = snippet.replace("\n", " ").strip()
        meta = info.get("meta", {})
        question = meta.get("question", "")
        query = meta.get("query", "")
        label = ""
        if doc_labels and url in doc_labels:
            label = f"[{doc_labels[url]}] "
        segments = []
        if url:
            segments.append(f"{label}**Title**: [{title}]({url})")
        else:
            segments.append(f"{label}**Title**: {title}")
        if query:
            segments.append(f"**Query**: `{query}`")
        if question:
            segments.append(f"**Question**: {question}")
        if snippet:
            segments.append(f"**Snippet**: {snippet}")
        items.append("; ".join(segments))
    return "\n".join(f"- {item}" for item in items)


def summarize_turn(
    turn: Dict,
    turn_number: int,
    doc_labels: Optional[Dict[str, str]] = None,
    citation_labels: Optional[Dict[int, str]] = None,
) -> str:
    role = turn.get("role", "Unknown")
    utype = turn.get("utterance_type", "")
    utter = (turn.get("utterance") or turn.get("raw_utterance") or "").strip()
    queries = turn.get("queries", [])
    docs = turn.get("raw_retrieved_info", [])
    if not (queries or docs or utter):
        return ""
    heading = f"### Turn {turn_number}: {role} ({utype})" if utype else f"### Turn {turn_number}: {role}"
    lines = [heading, ""]
    if queries:
        lines.append("**Queries**:")
        lines.extend(f"- `{q}`" for q in queries)
        lines.append("")
    retrieved_labels = []
    if docs:
        lines.append("**Documents**:")
        lines.append(format_documents(docs, doc_labels=doc_labels))
        lines.append("")
        if doc_labels:
            retrieved_labels = [
                doc_labels.get(info.get("url"))
                for info in docs
                if doc_labels.get(info.get("url"))
            ]
    if utter:
        lines.append("**Utterance**:")
        lines.append(utter)
        lines.append("")
    cited_labels = extract_cited_labels(utter, citation_labels)
    if retrieved_labels:
        lines.append(
            f"**Retrieved docs**: {', '.join(retrieved_labels)}"
        )
    if cited_labels:
        lines.append(f"**Cited docs**: {', '.join(cited_labels)}")
    if retrieved_labels:
        unused = [label for label in retrieved_labels if label not in cited_labels]
        if unused:
            lines.append(f"**Retrieved but uncited**: {', '.join(unused)}")
    if cited_labels or retrieved_labels:
        lines.append("")
    return "\n".join(lines)


def aggregate_documents(
    turns: List[Dict], extra_docs: Optional[List[Dict]] = None
) -> Dict[str, Dict]:
    doc_map: Dict[str, Dict] = {}

    def add_info(info: Dict):
        if not isinstance(info, dict):
            return
        url = info.get("url") or info.get("meta", {}).get("url")
        if not url or url in doc_map:
            return
        doc_map[url] = info

    for turn in turns or []:
        for info in turn.get("raw_retrieved_info", []) or []:
            add_info(info)

    if extra_docs:
        for info in extra_docs:
            add_info(info)

    return doc_map


def assign_document_labels(doc_map: Dict[str, Dict]) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for idx, url in enumerate(doc_map.keys(), start=1):
        if url:
            labels[url] = f"D{idx}"
    return labels


CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def extract_cited_labels(text: str, citation_to_label: Optional[Dict[int, str]]) -> List[str]:
    if not text or not citation_to_label:
        return []
    labels: List[str] = []
    for match in CITATION_PATTERN.findall(text):
        try:
            cid = int(match)
        except ValueError:
            continue
        label = citation_to_label.get(cid)
        if label and label not in labels:
            labels.append(label)
    return labels


def extract_conv_turn_numbers(log_data: Dict) -> List[int]:
    conv_nums: List[int] = []
    for stage_name in log_data.keys():
        stage_lower = stage_name.lower()
        match = re.search(r"conv turn:\s*(\d+)", stage_lower)
        if match:
            conv_nums.append(int(match.group(1)))
    return sorted(set(conv_nums))


def build_stage_structure_summary(
    log_data: Dict,
    warmstart_turn_numbers: List[int],
    interactive_turn_numbers: List[int],
    total_conversation_turns: int,
) -> List[str]:
    if not log_data:
        return []
    stage_names = list(log_data.keys())
    stage_order = " → ".join(stage_names)
    warmstart_label = (
        ", ".join(str(n) for n in warmstart_turn_numbers) if warmstart_turn_numbers else "None"
    )
    interactive_label = (
        ", ".join(str(n) for n in interactive_turn_numbers) if interactive_turn_numbers else "None"
    )
    lines = [
        f"- **Total pipeline stages**: {len(stage_names)}",
        f"- **Stage order**: {stage_order}",
        f"- **Warm-start turn numbers**: {warmstart_label}",
        f"- **Interactive turn numbers**: {interactive_label}",
        f"- **Total conversation turns logged**: {total_conversation_turns}",
    ]
    if warmstart_turn_numbers and interactive_turn_numbers:
        lines.append(
            "- Turns are numbered globally in the log, so interactive turns (e.g., "
            f"{interactive_label}) continue after the warm-start turns."
        )
    return lines


def summarize_doc_refs(
    docs: List[Dict],
    doc_labels: Optional[Dict[str, str]] = None,
    limit: int = 4,
) -> str:
    if not docs:
        return "_None_"
    entries = []
    for info in docs[:limit]:
        title = info.get("title") or info.get("description") or info.get("url", "Untitled")
        query = info.get("meta", {}).get("query")
        url = info.get("url")
        label_prefix = ""
        if doc_labels and url in doc_labels:
            label_prefix = f"[{doc_labels[url]}] "
        label = f"{label_prefix}{title}"
        if url:
            label = f"[{label}]({url})"
        if query:
            label = f"{label} (query `{query}`)"
        entries.append(label)
    if len(docs) > limit:
        entries.append(f"... +{len(docs) - limit} more")
    return "; ".join(entries)


def build_turn_lookup(conversation_history: List[Dict]) -> Dict[int, Dict]:
    lookup: Dict[int, Dict] = {}
    for idx, turn in enumerate(conversation_history or []):
        lookup[idx + 1] = turn
    return lookup


def extract_report_headings(report_markdown: str) -> List[str]:
    if not report_markdown:
        return []
    headings: List[str] = []
    for line in report_markdown.splitlines():
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            if heading:
                headings.append(heading)
    # Preserve order but remove duplicates
    seen = set()
    ordered = []
    for heading in headings:
        if heading not in seen:
            ordered.append(heading)
            seen.add(heading)
    return ordered


def summarize_lm_usage(log_data: Dict) -> str:
    sections = []
    for stage_name, stage_data in log_data.items():
        if not isinstance(stage_data, dict):
            continue
        lm_usage = stage_data.get("lm_usage")
        if not lm_usage:
            continue
        sections.append(f"### {stage_name}")
        for module_name, models in lm_usage.items():
            for model_name, usage in models.items():
                prompt = usage.get("prompt_tokens", 0)
                completion = usage.get("completion_tokens", 0)
                sections.append(
                    f"- `{module_name}` with `{model_name}`: prompt={prompt}, completion={completion}, total={prompt + completion}"
                )
        sections.append("")
    if not sections:
        return "_No LM usage data found._"
    return "\n".join(sections)


# Model role descriptions
MODEL_ROLE_DESCRIPTIONS = {
    "question_answering_lm": "Generates answers to questions using retrieved information",
    "discourse_manage_lm": "Manages conversation flow, selects experts, determines next turn",
    "question_asking_lm": "Generates questions for experts during warm start",
    "utterance_polishing_lm": "Polishes and refines generated utterances",
    "warmstart_outline_gen_lm": "Generates outline structure from warm start conversation",
    "knowledge_base_lm": "Transforms knowledge base sections into conversational format",
}

STAGE_DESCRIPTIONS = {
    "warm start stage": "Kick off background research, generate experts, and build initial mind map.",
    "warm start: outline generation": "Organize collected warm-start information into a report outline.",
    "warm start: insert into knowledge base": "Insert warm-start facts into the mind map.",
}


def describe_stage(stage_name: str) -> str:
    stage_lower = stage_name.lower()
    if stage_lower.startswith("conv turn"):
        match = re.search(r"conv turn:\s*(\d+)", stage_lower)
        turn_num = match.group(1) if match else "?"
        return (
            f"Interactive conversation turn {turn_num}: moderator selects the next speaker, "
            "runs optional retrieval, and captures the expert's answer."
        )
    if "report generation" in stage_lower:
        return "Synthesize the current knowledge base into the final report sections."
    for canonical, description in STAGE_DESCRIPTIONS.items():
        if canonical.lower() == stage_lower:
            return description
    return "Pipeline stage executed during the run."


def summarize_model_configs(lm_config: Dict, prompts_by_role: Dict[str, List[Dict]] = None) -> str:
    sections = []
    for role, cfg in lm_config.items():
        if not isinstance(cfg, dict):
            continue
        model = cfg.get("model", "(unspecified)")
        max_tokens = cfg.get("max_tokens", "N/A")
        prompt_count = len(prompts_by_role.get(role, [])) if prompts_by_role else 0
        description = MODEL_ROLE_DESCRIPTIONS.get(role, "Unknown role")
        sections.append(f"- `{role}` → model: `{model}`, max_tokens: {max_tokens}, prompts captured: {prompt_count}")
        sections.append(f"  - *Purpose*: {description}")
    if not sections:
        return "_No LM configuration data found._"
    return "\n".join(sections)


def extract_prompt_samples(
    log_data: Dict,
    max_stages: int = 4,
    max_per_stage: int = 2,
) -> str:
    sections = []
    stage_count = 0
    for stage_name, stage_data in log_data.items():
        if stage_count >= max_stages:
            break
        if not isinstance(stage_data, dict):
            continue
        lm_history = stage_data.get("lm_history")
        if not lm_history:
            continue
        stage_count += 1
        sections.append(f"### {stage_name}")
        for entry in lm_history[:max_per_stage]:
            prompt = (entry.get("prompt") or "").strip()
            response = (
                entry.get("response", {})
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            model = entry.get("response", {}).get("model", "unknown")
            sections.append(f"- **Model**: `{model}`")
            sections.append(f"  - **Prompt**: {prompt}")
            sections.append(f"  - **Response**: {response}")
        sections.append("")
    if not sections:
        return "_No prompt history captured in log file._"
    return "\n".join(sections)


def format_code_block(text: str, as_html: bool = False) -> str:
    if not text:
        return ""
    text = text.strip()
    if as_html:
        return f"<pre><code>{html.escape(text)}</code></pre>"
    return f"```\n{text}\n```"


def summarize_stage_timeline(log_data: Dict) -> str:
    if not log_data:
        return ""
    lines = ["| Stage | Duration (s) | LM Calls | Retrieval Queries |", "| --- | --- | --- | --- |"]
    for stage_name, stage_data in log_data.items():
        duration = round(stage_data.get("total_wall_time", 0.0), 2)
        lm_calls = len(stage_data.get("lm_history", []))
        queries = stage_data.get("query_count", 0)
        lines.append(f"| {stage_name} | {duration} | {lm_calls} | {queries} |")
    return "\n".join(lines)


def build_overview_section(
    topic: str,
    runner_args: Dict,
    warmstart_turns: List[Dict],
    interactive_turns: List[Dict],
    doc_catalog: Dict[str, Dict],
    log_data: Dict,
) -> List[str]:
    retrieval_calls = sum(stage.get("query_count", 0) for stage in log_data.values())
    lm_calls = sum(len(stage.get("lm_history", [])) for stage in log_data.values())
    total_docs = len(doc_catalog)
    total_turns = len(warmstart_turns) + len(interactive_turns)
    warmstart_count = len(warmstart_turns)
    interactive_count = len(interactive_turns)
    total_time = round(sum(stage.get("total_wall_time", 0.0) for stage in log_data.values()), 2)

    overview = [
        f"- **Topic**: {topic}",
        f"- **Warm start turns**: {warmstart_count}",
        f"- **Interactive turns**: {interactive_count}",
        f"- **Total turns**: {total_turns}",
        f"- **Retrieval queries issued**: {retrieval_calls}",
        f"- **LLM calls**: {lm_calls}",
        f"- **Unique documents added to knowledge base**: {total_docs}",
        f"- **Total pipeline time (s)**: {total_time}",
    ]
    return overview


def summarize_conversation_section(
    title: str,
    turns: List[Dict],
    start_index: int = 0,
    doc_labels: Optional[Dict[str, str]] = None,
    citation_labels: Optional[Dict[int, str]] = None,
    turn_numbers: Optional[List[int]] = None,
) -> str:
    if not turns:
        return ""
    section_lines = [f"## {title}", ""]
    content_found = False
    for idx, turn in enumerate(turns):
        if turn_numbers and idx < len(turn_numbers):
            turn_number = turn_numbers[idx]
        else:
            turn_number = start_index + idx + 1
        summary = summarize_turn(
            turn,
            turn_number,
            doc_labels=doc_labels,
            citation_labels=citation_labels,
        )
        if summary:
            section_lines.append(summary)
            content_found = True
    if not content_found:
        return ""
    return "\n".join(section_lines)


def shorten_text(text: str, limit: int = 220) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def format_turn_highlights(
    turn: Dict,
    doc_labels: Optional[Dict[str, str]] = None,
    citation_labels: Optional[Dict[int, str]] = None,
) -> List[str]:
    if not turn:
        return []
    lines = []
    role = turn.get("role", "Unknown")
    utype = turn.get("utterance_type", "").strip()
    role_line = f"- **Speaker**: {role}"
    if utype:
        role_line += f" ({utype})"
    lines.append(role_line)
    utter = shorten_text(turn.get("utterance") or turn.get("raw_utterance") or "", 400)
    if utter:
        lines.append(f"- **Utterance**: {utter}")
    queries = [q for q in (turn.get("queries") or []) if q]
    if queries:
        lines.append(
            f"- **Retrieval queries** (question_answering_lm): "
            + ", ".join(f"`{q}`" for q in queries)
        )
    else:
        lines.append("- **Retrieval queries**: _None_")
    docs = turn.get("raw_retrieved_info") or []
    if docs:
        lines.append(f"- **Documents referenced**: {summarize_doc_refs(docs, doc_labels=doc_labels)}")
        retrieved_labels = [
            doc_labels.get(info.get("url"))
            for info in docs
            if doc_labels and doc_labels.get(info.get("url"))
        ]
    else:
        retrieved_labels = []
    cited_labels = extract_cited_labels(turn.get("utterance"), citation_labels)
    if retrieved_labels:
        lines.append(f"- **Retrieved doc IDs**: {', '.join(retrieved_labels)}")
    if cited_labels:
        lines.append(f"- **Cited doc IDs**: {', '.join(cited_labels)}")
    if retrieved_labels:
        unused = [label for label in retrieved_labels if label not in cited_labels]
        if unused:
            lines.append(f"- **Unused retrieved docs**: {', '.join(unused)}")
    return lines


def format_warmstart_highlights(
    turns: List[Dict],
    doc_labels: Optional[Dict[str, str]] = None,
    citation_labels: Optional[Dict[int, str]] = None,
) -> List[str]:
    if not turns:
        return []
    lines = [f"- **Warm-start turns captured**: {len(turns)}"]
    queries = [q for turn in turns for q in (turn.get("queries") or []) if q]
    if queries:
        lines.append(f"- **Retrieval queries issued**: {', '.join(f'`{q}`' for q in queries)}")
    docs = [info for turn in turns for info in (turn.get("raw_retrieved_info") or [])]
    if docs:
        lines.append(f"- **Documents gathered**: {summarize_doc_refs(docs, doc_labels=doc_labels)}")
    sample_turns = turns[:2]
    for turn in sample_turns:
        speaker = turn.get("role", "Unknown")
        utter = shorten_text(turn.get("utterance") or "", 320)
        cited = extract_cited_labels(turn.get("utterance"), citation_labels)
        citation_text = f" (cites {', '.join(cited)})" if cited else ""
        lines.append(f"- **Sample – {speaker}**: {utter}{citation_text}")
    if len(turns) > len(sample_turns):
        lines.append(f"- ... plus {len(turns) - len(sample_turns)} more warm-start turns")
    return lines


def build_stage_flow(
    log_data: Dict,
    warmstart_turns: List[Dict],
    conversation_history: List[Dict],
    final_report_markdown: str,
    doc_labels: Optional[Dict[str, str]] = None,
    citation_labels: Optional[Dict[int, str]] = None,
) -> str:
    if not log_data:
        return ""
    lines: List[str] = []
    stage_items = list(log_data.items())
    turn_lookup = build_turn_lookup(conversation_history)
    report_headings = extract_report_headings(final_report_markdown)
    max_prompts_per_stage = 3
    conv_stage_counter = 0

    for idx, (stage_name, stage_data) in enumerate(stage_items, start=1):
        stage_lower = stage_name.lower()
        duration = round(stage_data.get("total_wall_time", 0.0), 2)
        lm_calls = len(stage_data.get("lm_history", []))
        queries = stage_data.get("query_count", 0)
        description = describe_stage(stage_name)
        if stage_lower.startswith("warm start"):
            stage_label = "Warm Start Research"
        elif stage_lower.startswith("conv turn"):
            conv_stage_counter += 1
            stage_label = f"Interactive Turn {conv_stage_counter} (log label: {stage_name})"
        elif "report generation" in stage_lower:
            stage_label = "Report Generation"
        else:
            stage_label = stage_name

        lines.append(f"### Stage {idx}: {stage_label}")
        lines.append(f"- **Purpose**: {description}")
        lines.append(f"- **Duration**: {duration} s")
        lines.append(f"- **LLM calls**: {lm_calls}")
        lines.append(f"- **Retrieval queries issued**: {queries}")
        llm_roles = []
        for role, models in (stage_data.get("lm_usage") or {}).items():
            for model_name in models.keys():
                llm_roles.append(f"{role} ({model_name})")
        if llm_roles:
            lines.append(f"- **LLMs invoked**: {', '.join(llm_roles)}")

        if stage_lower.startswith("warm start"):
            lines.extend(
                format_warmstart_highlights(
                    warmstart_turns, doc_labels=doc_labels, citation_labels=citation_labels
                )
            )
        elif stage_lower.startswith("conv turn"):
            match = re.search(r"conv turn:\s*(\d+)", stage_lower)
            if match:
                turn_num = int(match.group(1))
                turn_info = turn_lookup.get(turn_num)
                if turn_info:
                    lines.extend(
                        format_turn_highlights(
                            turn_info, doc_labels=doc_labels, citation_labels=citation_labels
                        )
                    )
        elif "report generation" in stage_lower and report_headings:
            lines.append(f"- **Sections produced**: {', '.join(report_headings)}")

        lines.append("")
        stage_prompts = stage_data.get("lm_history", [])[:max_prompts_per_stage]
        if stage_prompts:
            lines.append("**Key prompts & outputs:**")
            for entry in stage_prompts:
                prompt_text = (entry.get("prompt") or "").strip()
                response_text = (
                    entry.get("response", {})
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                role = identify_model_role(
                    prompt_text,
                    entry.get("response", {}).get("model", "unknown"),
                    stage_data.get("lm_usage", {}),
                    stage_name,
                )
                prompt_label = role or entry.get("response", {}).get("model", "unknown")
                lines.append(f"<details><summary>{prompt_label}</summary>")
                lines.append("**Prompt:**")
                lines.append(format_code_block(prompt_text))
                lines.append("")
                lines.append("**Output:**")
                lines.append(format_code_block(response_text))
                lines.append("</details>")
                lines.append("")

        next_stage = stage_items[idx][0] if idx < len(stage_items) else None
        if next_stage:
            lines.append(f"_Next → {next_stage}_")
        lines.append("")
    return "\n".join(lines)


EXPERT_PROMPT_MARKERS = [
    "select a group of diverse experts",
    "select a group of speakers",
    "roundtable discussion",
]


def extract_expert_prompts(log_data: Dict) -> str:
    matches = []
    for stage_name, stage_data in log_data.items():
        if not isinstance(stage_data, dict):
            continue
        for entry in stage_data.get("lm_history", []):
            prompt = entry.get("prompt") or ""
            if any(marker in prompt for marker in EXPERT_PROMPT_MARKERS):
                response = (
                    entry.get("response", {})
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                full_prompt = prompt.strip()
                full_response = response.strip()
                prompt_block = format_code_block(full_prompt)
                response_block = format_code_block(full_response)
                matches.append(
                    f"<details><summary>{html.escape(stage_name)}</summary>\n"
                    f"**Prompt**:\n{prompt_block}\n\n"
                    f"**Response**:\n{response_block}\n"
                    f"</details>"
                )
    if not matches:
        return "_No expert-generation prompts found in log history._"
    return "\n".join(matches)


# Prompt pattern mappings to identify which model role uses which prompts
PROMPT_PATTERNS = {
    "question_answering_lm": [
        "You are an expert who can use information effectively",
        "Make your response as informative as possible",
        "Gathered information:",
        "You want to provide insight on:",
        "Now give your response",
    ],
    "discourse_manage_lm": [
        "select a group of diverse experts",
        "select a group of speakers",
        "roundtable discussion",
        "Determine the next turn",
        "discourse policy",
        "You need to select a group",
    ],
    "question_asking_lm": [
        "You are a moderator in a roundtable discussion",
        "generate the next question",
        "Next question for the expert",
        "Topic for roundtable discussion",
        "Expert you are talking with",
    ],
    "utterance_polishing_lm": [
        "polish",
        "refine",
        "enhance the written",
        "improve the quality",
    ],
    "warmstart_outline_gen_lm": [
        "Generate a outline",
        "Generate an outline",
        "Write an outline",
        "wikipedia-like report",
        "conversation outline",
        "Write the conversation outline",
        "Draft outline you can reference",
        "Wikipedia page outline",
    ],
    "knowledge_base_lm": [
        "transform this section into",
        "engaging opening discussion",
        "section of a brief report",
        "section of a brief report",
        "transform this section",
        "engaging question",
    ],
}


def identify_model_role(prompt: str, model_name: str, lm_usage: Dict, stage_name: str = "") -> Optional[str]:
    """Try to identify which model role was used based on prompt patterns, lm_usage, and stage name."""
    prompt_lower = prompt.lower()
    stage_lower = stage_name.lower()
    
    # First, try pattern matching (most reliable)
    for role, patterns in PROMPT_PATTERNS.items():
        if any(pattern.lower() in prompt_lower for pattern in patterns):
            return role
    
    # Second, try stage name matching
    if "outline" in stage_lower and "generation" in stage_lower:
        return "warmstart_outline_gen_lm"
    elif "knowledge" in stage_lower or ("base" in stage_lower and "insert" not in stage_lower):
        return "knowledge_base_lm"
    elif "question" in stage_lower and "asking" in stage_lower:
        return "question_asking_lm"
    elif "answer" in stage_lower or "answering" in stage_lower:
        return "question_answering_lm"
    elif "expert" in stage_lower and "identify" in stage_lower:
        return "discourse_manage_lm"
    elif "polish" in stage_lower or "refine" in stage_lower:
        return "utterance_polishing_lm"
    
    # Third, if we have lm_usage, try to infer from usage patterns
    # If only one role was used in this stage, it's likely that one
    if lm_usage and len(lm_usage) == 1:
        return list(lm_usage.keys())[0]
    
    # Fourth, check if model name matches any role's expected model
    # This is less reliable but can help
    if lm_usage:
        # Count how many prompts each role has in lm_usage to help disambiguate
        for role, models in lm_usage.items():
            if model_name in models:
                # If this is the only role using this model in this stage, likely match
                other_roles_using_model = [
                    r for r, m in lm_usage.items() 
                    if r != role and model_name in m
                ]
                if not other_roles_using_model:
                    return role
    
    return None


def extract_prompts_by_model_role(log_data: Dict, lm_config: Dict) -> Dict[str, List[Dict]]:
    """Extract all prompts organized by model role."""
    prompts_by_role = defaultdict(list)
    
    for stage_name, stage_data in log_data.items():
        if not isinstance(stage_data, dict):
            continue
        
        lm_usage = stage_data.get("lm_usage", {})
        lm_history = stage_data.get("lm_history", [])
        
        for entry in lm_history:
            prompt = entry.get("prompt", "").strip()
            if not prompt:
                continue
            
            response_data = entry.get("response", {})
            model_name = response_data.get("model", "unknown")
            response_content = (
                response_data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            
            # Try to identify the model role
            role = identify_model_role(prompt, model_name, lm_usage, stage_name)
            if not role:
                role = "unknown"
            
            prompts_by_role[role].append({
                "stage": stage_name,
                "prompt": prompt,
                "response": response_content,
                "model": model_name,
            })
    
    return dict(prompts_by_role)


def format_prompts_for_role(role: str, prompts: List[Dict], max_prompts: Optional[int] = 5) -> str:
    """Format prompts for a specific model role."""
    if not prompts:
        return "_No prompts found._"
    
    lines = []
    prompts_to_show = prompts if max_prompts is None else prompts[:max_prompts]
    for idx, entry in enumerate(prompts_to_show):
        stage = entry.get("stage", "unknown")
        prompt = entry.get("prompt", "")
        response = entry.get("response", "")
        model = entry.get("model", "unknown")
        
        example_title = html.escape(f"Example {idx + 1} ({stage})")
        lines.append(f"<details><summary>{example_title}</summary>")
        lines.append(f"**Model**: `{model}`")
        lines.append("")
        lines.append("**Prompt**:")
        lines.append(format_code_block(prompt))
        lines.append("")
        lines.append("**Response**:")
        lines.append(format_code_block(response))
        lines.append("</details>")
        lines.append("")
    
    if max_prompts is not None and len(prompts) > max_prompts:
        lines.append(f"_... and {len(prompts) - max_prompts} more prompts_")
    
    return "\n".join(lines)


def markdown_to_html_with_collapsible(markdown: str) -> str:
    """Convert markdown to HTML with collapsible sections."""
    lines = markdown.split("\n")
    sections = []
    current_section = None
    current_content: List[str] = []
    in_code_block = False
    
    for line in lines:
        if line.startswith("```"):
            current_content.append(line)
            in_code_block = not in_code_block
            continue
        
        if in_code_block:
            current_content.append(line)
            continue
        
        if line.startswith("## ") and not line.startswith("###"):
            if current_section is not None:
                sections.append((current_section, current_content))
            elif current_content:
                sections.append((None, current_content))
            current_section = line[3:].strip()
            current_content = [line]
        else:
            current_content.append(line)
    
    if current_section is not None:
        sections.append((current_section, current_content))
    elif current_content:
        sections.append((None, current_content))
    
    # Second pass: convert to HTML with collapsible sections
    html_lines = []
    html_lines.append("""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Co-STORM Run Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #bdc3c7;
            padding-bottom: 5px;
        }
        h3 {
            color: #555;
            margin-top: 20px;
        }
        h4 {
            color: #666;
        }
        details {
            margin: 15px 0;
            padding: 10px;
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        summary {
            cursor: pointer;
            font-weight: bold;
            padding: 8px;
            background-color: #ecf0f1;
            border-radius: 3px;
            user-select: none;
            font-size: 1.1em;
        }
        summary:hover {
            background-color: #d5dbdb;
        }
        details[open] summary {
            background-color: #3498db;
            color: white;
        }
        pre {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            padding: 12px;
            overflow-x: auto;
            font-size: 0.9em;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        pre code {
            background-color: transparent;
            padding: 0;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
            background-color: white;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #3498db;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        a {
            color: #3498db;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        ul, ol {
            margin: 10px 0;
            padding-left: 30px;
        }
        li {
            margin: 5px 0;
        }
        p {
            margin: 10px 0;
        }
    </style>
</head>
<body>""")
    
    # Sections that should NOT be collapsible
    non_collapsible = ["Run Summary", "Runner Configuration", "Model Roles and Engines"]
    
    for section_title, content_lines in sections:
        if section_title and section_title not in non_collapsible:
            html_lines.append(f'<details><summary>{html.escape(section_title)}</summary>')
        else:
            html_lines.append("<div>")
        
        # Convert content to HTML
        in_code_block = False
        in_list = False
        
        for i, line in enumerate(content_lines):
            # Handle code blocks
            if line.startswith("```"):
                if in_code_block:
                    html_lines.append("</code></pre>")
                    in_code_block = False
                else:
                    lang = line[3:].strip()
                    html_lines.append(f'<pre><code class="language-{lang}">')
                    in_code_block = True
                continue
            
            if in_code_block:
                html_lines.append(html.escape(line))
                continue
            
            # Headers
            if line.startswith("# "):
                html_lines.append(f"<h1>{html.escape(line[2:])}</h1>")
            elif line.startswith("## "):
                html_lines.append(f"<h2>{html.escape(line[3:])}</h2>")
            elif line.startswith("### "):
                html_lines.append(f"<h3>{html.escape(line[4:])}</h3>")
            elif line.startswith("#### "):
                html_lines.append(f"<h4>{html.escape(line[5:])}</h4>")
            # Lists
            elif line.startswith("- ") or line.startswith("* "):
                if not in_list:
                    html_lines.append("<ul>")
                    in_list = True
                # Process markdown in list item
                item_text = line[2:]
                item_text = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'<a href="\2">\1</a>', item_text)
                item_text = re.sub(r'`([^`]+)`', r'<code>\1</code>', item_text)
                item_text = re.sub(r'\*\*([^\*]+)\*\*', r'<strong>\1</strong>', item_text)
                html_lines.append(f"<li>{item_text}</li>")
                if i == len(content_lines) - 1 or not (content_lines[i+1].startswith("- ") or content_lines[i+1].startswith("* ")):
                    html_lines.append("</ul>")
                    in_list = False
            # Tables
            elif line.startswith("|"):
                if i == 0 or not content_lines[i-1].startswith("|"):
                    html_lines.append("<table>")
                cells = [cell.strip() for cell in line.split("|")[1:-1]]
                if "---" in line:
                    html_lines.append("</thead><tbody>")
                else:
                    is_header = i == 0 or (i > 1 and "---" in content_lines[i-1])
                    html_lines.append("<tr>")
                    for cell in cells:
                        tag = "th" if is_header else "td"
                        html_lines.append(f"<{tag}>{html.escape(cell)}</{tag}>")
                    html_lines.append("</tr>")
                if i == len(content_lines) - 1 or not content_lines[i+1].startswith("|"):
                    html_lines.append("</tbody></table>")
            # Empty lines
            elif line.strip() == "":
                if not in_list:
                    html_lines.append("<br>")
            # Regular text
            else:
                if not in_list:
                    stripped = line.strip()
                    raw_html_prefixes = (
                        "<details",
                        "</details",
                        "<summary",
                        "</summary",
                        "<pre",
                        "</pre",
                        "<code",
                        "</code",
                    )
                    if any(stripped.startswith(prefix) for prefix in raw_html_prefixes):
                        html_lines.append(line)
                        continue
                    # Convert markdown links
                    line = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'<a href="\2">\1</a>', line)
                    # Convert inline code
                    line = re.sub(r'`([^`]+)`', r'<code>\1</code>', line)
                    # Convert bold
                    line = re.sub(r'\*\*([^\*]+)\*\*', r'<strong>\1</strong>', line)
                    # Convert italic
                    line = re.sub(r'\*([^\*]+)\*', r'<em>\1</em>', line)
                    html_lines.append(f"<p>{line}</p>")
        
        if in_code_block:
            html_lines.append("</code></pre>")
        
        if section_title and section_title not in non_collapsible:
            html_lines.append("</details>")
        else:
            html_lines.append("</div>")
    
    html_lines.append("</body></html>")
    return "\n".join(html_lines)


def build_report(run_dir: Path) -> str:
    log_path = run_dir / "log.json"
    instance_path = run_dir / "instance_dump.json"
    if not log_path.exists() or not instance_path.exists():
        raise FileNotFoundError("log.json and/or instance_dump.json not found in run directory")

    log_data = load_json(log_path)
    instance_data = load_json(instance_path)

    topic = instance_data.get("runner_argument", {}).get("topic", "(unknown topic)")
    runner_args = instance_data.get("runner_argument", {})
    warmstart_archive_turns = instance_data.get("warmstart_conv_archive", []) or []
    conv_history = instance_data.get("conversation_history", []) or []
    lm_config = instance_data.get("lm_config", {})
    knowledge_base = instance_data.get("knowledge_base", {})
    kb_info_map = (knowledge_base or {}).get("info_uuid_to_info_dict", {}) or {}
    kb_docs = list(kb_info_map.values())

    turn_lookup = build_turn_lookup(conv_history)
    total_conversation_turns = len(turn_lookup)
    conv_turn_numbers = extract_conv_turn_numbers(log_data)
    if conv_turn_numbers:
        first_conv_turn = min(conv_turn_numbers)
        warmstart_turn_numbers = [
            idx for idx in sorted(turn_lookup.keys()) if idx < first_conv_turn
        ]
    else:
        warmstart_turn_numbers = sorted(turn_lookup.keys())
    warmstart_turns = [turn_lookup[idx] for idx in warmstart_turn_numbers if idx in turn_lookup]
    interactive_turns = [turn_lookup[idx] for idx in conv_turn_numbers if idx in turn_lookup]

    all_turns = (warmstart_archive_turns or []) + conv_history
    doc_catalog = aggregate_documents(all_turns, kb_docs)
    doc_labels = assign_document_labels(doc_catalog)
    citation_to_doc_label: Dict[int, str] = {}
    for info in kb_info_map.values():
        citation_id = info.get("citation_uuid")
        url = info.get("url")
        label = doc_labels.get(url)
        if isinstance(citation_id, int) and citation_id != -1 and label:
            citation_to_doc_label[citation_id] = label
    report_md_path = run_dir / "report.md"
    final_report_content = ""
    if report_md_path.exists():
        final_report_content = report_md_path.read_text(encoding="utf-8").strip()

    # Extract prompts organized by model role
    prompts_by_role = extract_prompts_by_model_role(log_data, lm_config)

    md_lines = [f"# Run Summary: {topic}", ""]

    md_lines.append("## Run Overview")
    md_lines.extend(
        build_overview_section(
            topic,
            runner_args,
            warmstart_turns,
            interactive_turns,
            doc_catalog,
            log_data,
        )
    )
    md_lines.append("")

    stage_structure = build_stage_structure_summary(
        log_data,
        warmstart_turn_numbers,
        conv_turn_numbers,
        total_conversation_turns,
    )
    if stage_structure:
        md_lines.append("## Pipeline Structure")
        md_lines.extend(stage_structure)
        md_lines.append("")

    md_lines.append("## Runner Configuration")
    md_lines.append(format_kv_table(runner_args))
    md_lines.append("")

    md_lines.append("## Model Roles and Engines")
    md_lines.append(summarize_model_configs(lm_config, prompts_by_role))
    md_lines.append("")

    stage_flow = build_stage_flow(
        log_data,
        warmstart_turns,
        conv_history,
        final_report_content,
        doc_labels,
        citation_to_doc_label,
    )
    if stage_flow:
        md_lines.append("## Stage-by-Stage Walkthrough")
        md_lines.append(stage_flow)
        md_lines.append("")

    # Stage timeline summary
    timeline = summarize_stage_timeline(log_data)
    if timeline:
        md_lines.append("## Stage Timeline")
        md_lines.append(timeline)
        md_lines.append("")

    warmstart_section = summarize_conversation_section(
        "Warm Start Conversation",
        warmstart_turns,
        start_index=0,
        doc_labels=doc_labels,
        citation_labels=citation_to_doc_label,
        turn_numbers=warmstart_turn_numbers,
    )
    if warmstart_section:
        md_lines.append(warmstart_section)
        md_lines.append("")

    interactive_section = summarize_conversation_section(
        "Interactive Conversation",
        interactive_turns,
        start_index=len(warmstart_turns),
        doc_labels=doc_labels,
        citation_labels=citation_to_doc_label,
        turn_numbers=conv_turn_numbers,
    )
    if interactive_section:
        md_lines.append(interactive_section)
        md_lines.append("")

    if doc_catalog:
        md_lines.append("## Document Catalog")
        for url, info in doc_catalog.items():
            title = info.get("title") or info.get("description", url)
            label = doc_labels.get(url)
            label_prefix = f"[{label}] " if label else ""
            query = info.get("meta", {}).get("query")
            question = info.get("meta", {}).get("question")
            snippet = (info.get("snippets") or [""])[0].replace("\n", " ").strip()
            line = f"- {label_prefix}[{title}]({url})"
            if query:
                line += f"; **Query**: `{query}`"
            if question:
                line += f"; **Question**: {question}"
            if snippet:
                line += f"; **Snippet**: {snippet}"
            md_lines.append(line)
        md_lines.append("")

    if final_report_content:
        md_lines.append("## Final Report Output")
        md_lines.append(final_report_content)
        md_lines.append("")

    md_lines.append("## LLM Usage Summary")
    md_lines.append(summarize_lm_usage(log_data))
    md_lines.append("")

    markdown = "\n".join(md_lines)
    html_output = markdown_to_html_with_collapsible(markdown)
    return html_output


def main():
    parser = argparse.ArgumentParser(description="Generate HTML run summary from a Co-STORM run directory")
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to run directory containing log.json and instance_dump.json",
    )
    args = parser.parse_args()

    html_output = build_report(args.run_dir)

    html_path = args.run_dir / "run_report.html"
    html_path.write_text(html_output, encoding="utf-8")

    print(f"HTML summary written to {html_path}")


def run():
    main()


if __name__ == "__main__":
    run()
