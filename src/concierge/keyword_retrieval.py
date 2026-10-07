"""Local BM25 retrieval for workspaces without an embeddings provider."""

from collections import Counter
from math import log
import re

from langchain_core.documents import Document

STOP_WORDS = frozenset(
    "a an and are as at be by for from how i in is it me my of on or the to what when with".split()
)


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [word for word in words if word not in STOP_WORDS]


def search_documents(documents: list[Document], query: str, k: int) -> list[Document]:
    """Return up to k matching chunks, retaining their citation metadata."""
    query_terms = set(_tokens(query))
    if not query_terms or k <= 0 or not documents:
        return []

    term_counts = []
    for document in documents:
        searchable_text = document.page_content + " " + document.metadata.get("topic", "")
        term_counts.append(Counter(_tokens(searchable_text)))
    lengths = [sum(counts.values()) for counts in term_counts]
    average_length = sum(lengths) / len(lengths) or 1
    document_frequency = Counter(term for counts in term_counts for term in counts)

    ranked = []
    for document, counts, length in zip(documents, term_counts, lengths):
        score = 0.0
        for term in query_terms:
            frequency = counts[term]
            if not frequency:
                continue
            matching_documents = document_frequency[term]
            inverse_frequency = log(
                1 + (len(documents) - matching_documents + 0.5) / (matching_documents + 0.5)
            )
            length_penalty = 1.5 * (0.25 + 0.75 * length / average_length)
            score += inverse_frequency * frequency * 2.5 / (frequency + length_penalty)
        if score > 0:
            ranked.append((score, document))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [document for _, document in ranked[:k]]
