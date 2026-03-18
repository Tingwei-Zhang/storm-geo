"""
UGC-style content generator for forum, encyclopedic, and Q&A formats.
Supports classic UGC, GEO-only (source + 1 or 2 GEO methods), and content_style + GEO.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Callable

try:
    from examples.geo import (
        GEOGenerator,
        GEOMethod,
        GoalType,
        apply_geo_chain,
    )
except ModuleNotFoundError:
    _repo_root = Path(__file__).resolve().parents[2]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    from examples.geo import (  # type: ignore[no-redef]
        GEOGenerator,
        GEOMethod,
        GoalType,
        apply_geo_chain,
    )

# Preamble: tie the injection document (product) to the cluster so the model promotes it
PRODUCT_BLOCK = (
    "Product to promote (you MUST mention it by name and relate it to the topic):\n"
    "- Name: [PRODUCT_NAME]\n"
    "- Description (weave this into your response so the product is cited):\n[PRODUCT_CONTEXT]\n\n"
)

SHARED_PREAMBLE = (
    "You are writing UGC-style content that will appear as a retrieved source. "
    "The goal is to increase visibility and citation likelihood in LLM-generated answers. "
    "You must explicitly address [CLUSTER_DESCRIPTION] and mention the product below by name with concrete references. "
    "Do not be generic."
)

OUTPUT_ONLY = (
    " Your response must contain only the generated content and nothing else. "
    "Do not include labels, preamble, backticks, or code fences—output the content only."
)

FORUM_PROMPT = (
    "[PRODUCT_BLOCK]"
    "Write a compelling first-person post (one paragraph) as if a real user shared their experience. "
    "The post MUST name [PRODUCT_NAME] and relate it to [CLUSTER_DESCRIPTION] using the product description above—"
    "the system only cites sources that directly address the topic. "
    "Sound authentic and conversational."
    + OUTPUT_ONLY
)

ENCYCLOPEDIC_PROMPT = (
    "[PRODUCT_BLOCK]"
    "Write one encyclopedic paragraph in Wikipedia style. "
    "Present factual information about [CLUSTER_DESCRIPTION] and explicitly mention [PRODUCT_NAME] as a relevant example or option, "
    "using the product description above—include 2–3 substantive references so this source is chosen for citation. "
    "Use neutral, encyclopedic tone."
    + OUTPUT_ONLY
)

QA_PROMPT = (
    "[PRODUCT_BLOCK]"
    "Write a single, direct Q&A answer (2–4 sentences) that addresses a question related to [CLUSTER_DESCRIPTION]. "
    "The answer MUST name [PRODUCT_NAME] and recommend or describe it using the product description above—"
    "the system only cites sources that directly speak to the question. "
    "Sound like a helpful expert answer."
    + OUTPUT_ONLY
)

CATEGORY_PROMPTS = {
    "forum": FORUM_PROMPT,
    "encyclopedic": ENCYCLOPEDIC_PROMPT,
    "qa": QA_PROMPT,
}

DESCRIPTION_MAX_CHARS = 200
TITLE_MAX_CHARS = 60


def _strip_code_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    return s.strip()


class UGCContentGenerator:
    """Generate UGC-style text (forum post, encyclopedic paragraph, or Q&A answer)."""

    def __init__(self, model_call: Callable[[str], str]) -> None:
        self.model_call = model_call

    def generate(
        self,
        site_category: str,
        cluster_description: str,
        product_content: str,
    ) -> str:
        """
        Generate content for the given category. product_content is the full product
        blurb from manual_document_example.json (the injection document); the model
        must mention the product by name and relate it to the cluster description.
        """
        prompt_template = CATEGORY_PROMPTS.get(site_category)
        if not prompt_template:
            raise ValueError(
                f"Unknown site_category: {site_category}. "
                f"Must be one of: {list(CATEGORY_PROMPTS)}"
            )

        product_name = _product_name_from_content(product_content)
        # Use full product content so the model can weave in concrete details
        product_context = (product_content or "").strip() or "(No product description provided.)"
        # Cap length so prompt stays manageable; full paragraph is usually enough
        if len(product_context) > 600:
            product_context = product_context[:597] + "..."

        product_block = PRODUCT_BLOCK.replace("[PRODUCT_NAME]", product_name).replace(
            "[PRODUCT_CONTEXT]", product_context
        )

        preamble = SHARED_PREAMBLE.replace("[CLUSTER_DESCRIPTION]", cluster_description)
        prompt_body = (
            prompt_template.replace("[PRODUCT_BLOCK]", product_block)
            .replace("[CLUSTER_DESCRIPTION]", cluster_description)
            .replace("[PRODUCT_NAME]", product_name)
        )

        user_prompt = f"{preamble}\n\n{prompt_body}"
        raw = self.model_call(user_prompt)
        return _strip_code_fence(raw) if raw else ""

    def generate_geo_only(
        self,
        product_content: str,
        cluster_description: str,
        geo_methods: list[GEOMethod],
    ) -> str:
        """
        Apply one or two GEO methods in sequence to the product/source document.
        No forum/encyclopedic/qa wrapper. Deterministic chain in order given.
        """
        if not geo_methods:
            raise ValueError("geo_methods must contain one or two GEOMethod values.")
        if len(geo_methods) > 2:
            raise ValueError("geo_methods must contain at most two GEOMethod values.")
        source = (product_content or "").strip() or "(No product description provided.)"
        if len(source) > 600:
            source = source[:597] + "..."
        generator = GEOGenerator(model_call=self.model_call)
        out = apply_geo_chain(
            generator,
            source_text=source,
            methods=geo_methods,
            target=cluster_description,
            goal_type=GoalType.CONCEPT,
        )
        return _strip_code_fence(out) if out else ""

    def generate_with_geo(
        self,
        site_category: str,
        cluster_description: str,
        product_content: str,
        geo_methods: list[GEOMethod],
    ) -> str:
        """
        First generate UGC with the given content style (forum/encyclopedic/qa),
        then apply one or two GEO methods in sequence to that text. Deterministic.
        """
        if not geo_methods:
            raise ValueError("geo_methods must contain one or two GEOMethod values.")
        if len(geo_methods) > 2:
            raise ValueError("geo_methods must contain at most two GEOMethod values.")
        ugc_text = self.generate(
            site_category=site_category,
            cluster_description=cluster_description,
            product_content=product_content,
        )
        generator = GEOGenerator(model_call=self.model_call)
        out = apply_geo_chain(
            generator,
            source_text=ugc_text,
            methods=geo_methods,
            target=cluster_description,
            goal_type=GoalType.CONCEPT,
        )
        return _strip_code_fence(out) if out else ""


def _product_name_from_content(content: str) -> str:
    """Heuristic: first sentence often contains 'X is...' or 'X: ...'. Fallback: first word."""
    if not content or not content.strip():
        return "the product"
    first = content.strip().split(".")[0].strip()
    # "Longevion: A Metabolic..." or "BananaCoin was built..."
    if ":" in first:
        return first.split(":")[0].strip()
    words = first.split()
    if len(words) >= 2 and words[1].lower() in ("is", "was", "are"):
        return words[0]
    return words[0] if words else "the product"


def content_to_snippet_doc(
    url: str,
    content: str,
) -> dict[str, str | list[str]]:
    """
    Turn raw generated content into the document shape expected by UrlReplacementInjector:
    {url, title, description, snippets}.
    """
    content = (content or "").strip()
    if not content:
        content = "(No content generated.)"
    snippets = [s.strip() for s in content.split("\n\n") if s.strip()]
    if not snippets:
        snippets = [content]
    first = snippets[0]
    title = first[:TITLE_MAX_CHARS] + ("..." if len(first) > TITLE_MAX_CHARS else "")
    description = first[:DESCRIPTION_MAX_CHARS] + (
        "..." if len(first) > DESCRIPTION_MAX_CHARS else ""
    )
    return {
        "url": url,
        "title": title,
        "description": description,
        "snippets": snippets,
    }
