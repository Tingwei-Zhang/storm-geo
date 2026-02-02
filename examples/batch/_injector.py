"""
FirstRetrievalInjector for batch runs: inject manual documents into the first retrieval only.
Same behavior as examples.manual_examples.run_costorm_demo_first_retrieval_manual.FirstRetrievalInjector.
"""

from typing import List

from knowledge_storm.interface import Information


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
            return manual_results + results
        return results

    __call__ = forward
