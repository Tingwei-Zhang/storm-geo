"""
UrlReplacementInjector: replace retrieved document content with injected UGC when URL matches.
Keeps the original URL for citation authenticity; swaps title, description, and snippets.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse


def _url_base(url: str) -> str:
    """Extract url_base (normalized host) from a full URL for matching."""
    if not url or not url.strip():
        return ""
    parsed = urlparse(url.strip())
    netloc = (parsed.netloc or "").strip().lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


class UrlReplacementInjector:
    """
    Wraps a retriever. After each retrieval, replaces result(s) whose url
    is in replacement_map with the injected document (same url, new
    title/description/snippets).

    If replace_first_only is True (general UGC injection mode), only the
    first result whose URL *base* (host) matches a base in replacement_map
    is replaced; the match is by url_base only, not exact URL. All other
    results are left as returned. If False (default), every result whose
    exact URL is in replacement_map is replaced.
    """

    def __init__(
        self,
        base_retriever: Any,
        replacement_map: dict[str, dict],
        replace_first_only: bool = False,
    ) -> None:
        self.base_retriever = base_retriever
        self.replacement_map = replacement_map
        self.replace_first_only = replace_first_only
        # url_base -> one injected doc (first URL we have for that base)
        self._by_base: dict[str, dict] = {}
        if replacement_map:
            for full_url, doc in replacement_map.items():
                base = _url_base(full_url)
                if base and base not in self._by_base:
                    self._by_base[base] = doc
        if hasattr(base_retriever, "k"):
            self.k = base_retriever.k

    def forward(self, query_or_queries, exclude_urls=None):
        results = self.base_retriever.forward(
            query_or_queries=query_or_queries, exclude_urls=exclude_urls
        )
        if not self.replacement_map:
            return results
        out = []
        replaced_once = False
        for r in results:
            if not isinstance(r, dict):
                out.append(r)
                continue
            url = (r.get("url") or "").strip()
            if self.replace_first_only:
                base = _url_base(url)
                if base in self._by_base and not replaced_once:
                    inj = self._by_base[base]
                    snippets = (
                        inj.get("snippets")
                        if "snippets" in inj
                        else r.get("snippets") or []
                    )
                    out.append({
                        "url": url,
                        "title": inj.get("title") or r.get("title") or "",
                        "description": (
                            inj.get("description") or r.get("description") or ""
                        ),
                        "snippets": snippets,
                    })
                    replaced_once = True
                else:
                    out.append(r)
            else:
                if url in self.replacement_map:
                    inj = self.replacement_map[url]
                    snippets = (
                        inj.get("snippets")
                        if "snippets" in inj
                        else r.get("snippets") or []
                    )
                    out.append({
                        "url": url,
                        "title": inj.get("title") or r.get("title") or "",
                        "description": (
                            inj.get("description") or r.get("description") or ""
                        ),
                        "snippets": snippets,
                    })
                else:
                    out.append(r)
        return out

    __call__ = forward
