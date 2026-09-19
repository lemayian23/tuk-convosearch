"""
PubMedStore — fetches abstracts from NCBI E-utilities, embeds them,
and stores in a local FAISS index.

Extends BaseVectorStore so it drops into any pipeline unchanged.

Location: app/rag/stores/pubmed_store.py
"""

import os
import time
import pickle
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from app.rag.stores.base_store import BaseVectorStore


# NCBI E-utilities endpoints
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# NCBI asks for <= 3 requests/sec without an API key
REQUEST_DELAY_SECONDS = 0.4


class PubMedStore(BaseVectorStore):
    def __init__(
        self,
        persist_directory: str = "./pubmed_index",
        embedding_model: str = "all-MiniLM-L6-v2",
        dimension: int = 384,
        email: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Args:
            persist_directory: where FAISS index + metadata live
            embedding_model:   sentence-transformers model name
            dimension:         embedding size (must match model)
            email:             NCBI recommends including your email
            api_key:           optional NCBI API key (10 req/sec instead of 3)
        """
        self.persist_directory = persist_directory
        self.index_file = os.path.join(persist_directory, "faiss_index.bin")
        self.metadata_file = os.path.join(persist_directory, "metadata.pkl")
        self.dimension = dimension
        self.email = email
        self.api_key = api_key
        self.request_delay = 0.1 if api_key else REQUEST_DELAY_SECONDS

        os.makedirs(persist_directory, exist_ok=True)

        print("  [PubMedStore] Loading embedding model...")
        self.embedder = SentenceTransformer(embedding_model)

        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            print(f"  [PubMedStore] Loading existing index from {self.index_file}")
            self.index = faiss.read_index(self.index_file)
            with open(self.metadata_file, "rb") as f:
                self.metadata = pickle.load(f)
            print(f"  [PubMedStore] Loaded {len(self.metadata)} chunks")
        else:
            print("  [PubMedStore] Creating new index (IndexFlatL2)")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = []
            print("  [PubMedStore] New index created")

    # ------------------------------------------------------------------ #
    # Public: fetch from NCBI + index
    # ------------------------------------------------------------------ #

    def fetch_and_index(
        self,
        query: str,
        max_results: int = 20,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        skip_existing: bool = True,
    ) -> int:
        """
        Search PubMed, fetch abstracts, chunk, embed, and add to the index.

        Returns:
            Number of chunks added.
        """
        print(f"\n  [PubMedStore] Searching PubMed for: {query!r}")
        pmids = self._esearch(query, max_results=max_results)
        print(f"  [PubMedStore] Found {len(pmids)} PMIDs")

        if not pmids:
            return 0

        if skip_existing:
            existing_pmids = {
                m["metadata"].get("pmid") for m in self.metadata
            }
            pmids = [p for p in pmids if p not in existing_pmids]
            print(f"  [PubMedStore] {len(pmids)} new PMIDs after dedup")

        if not pmids:
            print("  [PubMedStore] Nothing new to fetch")
            return 0

        articles = self._efetch(pmids)
        print(f"  [PubMedStore] Fetched {len(articles)} article records")

        chunks = self._chunk_articles(articles, chunk_size, chunk_overlap)
        print(f"  [PubMedStore] Created {len(chunks)} chunks")

        if not chunks:
            return 0

        return self.add_chunks(chunks)

    # ------------------------------------------------------------------ #
    # BaseVectorStore API
    # ------------------------------------------------------------------ #

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=False)
        self.index.add(np.array(embeddings).astype("float32"))

        for chunk in chunks:
            self.metadata.append({
                "id": len(self.metadata),
                "text": chunk["text"],
                "metadata": chunk["metadata"],
            })

        self._save()
        print(f"  [PubMedStore] Added {len(chunks)} chunks. Total: {self.index.ntotal}")
        return len(chunks)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            print("  [PubMedStore] Index empty — no results")
            return []

        k = min(k, self.index.ntotal)
        query_embedding = self.embedder.encode([query], show_progress_bar=False)
        distances, indices = self.index.search(
            np.array(query_embedding).astype("float32"), k
        )

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            chunk = self.metadata[idx]
            distance = float(distances[0][i])
            results.append({
                "text": str(chunk["text"]),
                "metadata": {
                    "source": str(chunk["metadata"].get("source", "pubmed")),
                    "pmid": str(chunk["metadata"].get("pmid", "")),
                    "title": str(chunk["metadata"].get("title", "")),
                    "journal": str(chunk["metadata"].get("journal", "")),
                    "year": str(chunk["metadata"].get("year", "")),
                    "authors": str(chunk["metadata"].get("authors", "")),
                    "chunk_index": int(chunk["metadata"].get("chunk_index", 0)),
                },
                "distance": distance,
                "relevance_score": float(1 / (1 + distance)),
            })

        print(f"  [PubMedStore] Found {len(results)} chunks")
        return results

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_chunks": int(self.index.ntotal),
            "dimension": int(self.dimension),
            "persist_directory": str(self.persist_directory),
            "index_type": "IndexFlatL2",
            "store_type": "PubMedStore",
        }

    def clear_all(self) -> None:
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        self._save()
        print("  [PubMedStore] Index cleared")

    # ------------------------------------------------------------------ #
    # NCBI E-utilities
    # ------------------------------------------------------------------ #

    def _esearch(self, query: str, max_results: int) -> List[str]:
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        }
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            r = requests.get(ESEARCH_URL, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            return data.get("esearchresult", {}).get("idlist", []) or []
        except Exception as e:
            print(f"  [PubMedStore] esearch failed: {e}")
            return []

    def _efetch(self, pmids: List[str]) -> List[Dict[str, str]]:
        if not pmids:
            return []

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            time.sleep(self.request_delay)
            r = requests.get(EFETCH_URL, params=params, timeout=30)
            r.raise_for_status()
        except Exception as e:
            print(f"  [PubMedStore] efetch failed: {e}")
            return []

        return self._parse_pubmed_xml(r.text)

    @staticmethod
    def _parse_pubmed_xml(xml_text: str) -> List[Dict[str, str]]:
        articles = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            print(f"  [PubMedStore] XML parse error: {e}")
            return []

        for article in root.findall(".//PubmedArticle"):
            try:
                pmid = article.findtext(".//PMID", default="").strip()
                title = article.findtext(".//ArticleTitle", default="").strip()
                journal = article.findtext(".//Journal/Title", default="").strip()
                year = (
                    article.findtext(".//PubDate/Year", default="")
                    or article.findtext(".//PubDate/MedlineDate", default="")
                ).strip()[:4]

                abstract_parts = [
                    (el.text or "").strip()
                    for el in article.findall(".//Abstract/AbstractText")
                ]
                abstract = " ".join(p for p in abstract_parts if p)

                authors = []
                for author in article.findall(".//Author")[:3]:
                    last = author.findtext("LastName", default="").strip()
                    init = author.findtext("Initials", default="").strip()
                    if last:
                        authors.append(f"{last} {init}".strip())
                authors_str = ", ".join(authors)

                if abstract:
                    articles.append({
                        "pmid": pmid,
                        "title": title,
                        "journal": journal,
                        "year": year,
                        "authors": authors_str,
                        "abstract": abstract,
                    })
            except Exception as e:
                print(f"  [PubMedStore] Skipping malformed article: {e}")
                continue

        return articles

    # ------------------------------------------------------------------ #
    # Chunking
    # ------------------------------------------------------------------ #

    @staticmethod
    def _chunk_articles(
        articles: List[Dict[str, str]],
        chunk_size: int,
        chunk_overlap: int,
    ) -> List[Dict[str, Any]]:
        chunks = []
        for art in articles:
            text = f"{art['title']}\n\n{art['abstract']}".strip()
            if not text:
                continue

            source_label = (
                f"PMID:{art['pmid']} — {art['title'][:80]}"
                if art["pmid"]
                else art["title"][:80]
            )

            parts = PubMedStore._split_text(text, chunk_size, chunk_overlap)
            for i, part in enumerate(parts):
                chunks.append({
                    "text": part,
                    "metadata": {
                        "source": source_label,
                        "pmid": art["pmid"],
                        "title": art["title"],
                        "journal": art["journal"],
                        "year": art["year"],
                        "authors": art["authors"],
                        "chunk_index": i,
                    },
                })
        return chunks

    @staticmethod
    def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
            if start >= len(text):
                break
        return chunks

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _save(self) -> None:
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, "wb") as f:
            pickle.dump(self.metadata, f)