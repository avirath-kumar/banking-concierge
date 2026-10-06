"""Vector-store retrieval over the synthetic banking knowledge base."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

KB_DIR = Path(__file__).parent / "kb"


def _make_embeddings() -> OpenAIEmbeddings:
    """Create an OpenAIEmbeddings instance."""
    base_url = os.getenv("BASE_URL")
    if base_url:
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            base_url=base_url,
            api_key=os.getenv("CONCIERGE_GATEWAY_API_KEY") or os.environ["LANGSMITH_API_KEY"],
        )
    return OpenAIEmbeddings(model="text-embedding-3-small")


def _load_kb_documents() -> list[Document]:
    docs: list[Document] = []
    for md_path in sorted(KB_DIR.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        docs.append(
            Document(
                page_content=text,
                metadata={"source": md_path.name, "topic": md_path.stem},
            )
        )
    return docs


@lru_cache(maxsize=1)
def get_vector_store() -> InMemoryVectorStore:
    docs = _load_kb_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    embeddings = _make_embeddings()
    return InMemoryVectorStore.from_documents(chunks, embeddings)


def retrieve(query: str, k: int = 4) -> list[Document]:
    return get_vector_store().similarity_search(query, k=k)
