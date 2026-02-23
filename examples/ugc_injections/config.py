"""
UGC (user-generated content) site categories for URL replacement injection.
Maps url_base from recurring_urls_raw.csv to content style: forum, encyclopedic, or qa.
"""

UGC_SITE_CATEGORIES = {
    "forum": ["reddit.com"],
    "encyclopedic": ["en.wikipedia.org"],
    "qa": [
        "law.stackexchange.com",
        "money.stackexchange.com",
        "quora.com",
        "avvo.com",
        "answers.justia.com",
    ],
}


def get_ugc_url_bases() -> set[str]:
    """All url_bases that are UGC (any category)."""
    bases = set()
    for bases_list in UGC_SITE_CATEGORIES.values():
        bases.update(bases_list)
    return bases


def url_base_to_category(url_base: str) -> str | None:
    """Return category for url_base, or None if not UGC."""
    url_base = (url_base or "").strip().lower()
    for category, bases in UGC_SITE_CATEGORIES.items():
        if url_base in bases:
            return category
    return None
