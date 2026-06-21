"""
Rebuild FAISS index from all documents in docs folder
Location: backend/rebuild_faiss.py

Can be run two ways:
  1. As a standalone script:  python rebuild_faiss.py
  2. Imported and called directly from the running server
     (see app/api/admin.py), which avoids spawning a second OS process
     that would otherwise contend with the live server for the SQLite
     database file.
"""

import os
import sys

# Ensure this works whether run as a script (python rebuild_faiss.py)
# or imported as a module (from rebuild_faiss import rebuild_index)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.document_loader import DocumentLoader
from app.services.chunking import DocumentChunker
from app.services.faiss_vector_store import FAISSVectorStore
from app.services import database


def rebuild_index(docs_folder: str = None, verbose: bool = True) -> dict:
    """
    Core rebuild logic, callable directly from Python (no subprocess needed).

    Args:
        docs_folder: path to the folder containing source documents.
                     Defaults to '../docs' relative to this file, matching
                     the original script's behaviour.
        verbose: if True, prints progress exactly as the original CLI script did.

    Returns:
        dict with summary stats: total_chunks, total_documents, dimension, index_type
    """
    def log(msg):
        if verbose:
            print(msg)

    if docs_folder is None:
        docs_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")

    log("=" * 60)
    log("Rebuilding FAISS Index from All Documents")
    log("=" * 60)

    database.init_db()

    log("\n📁 Loading documents from docs folder...")
    loader = DocumentLoader()
    chunker = DocumentChunker()

    documents = loader.load_documents_from_folder(docs_folder)
    log(f"Loaded {len(documents)} documents")

    if not documents:
        log("No documents found!")
        return {"total_chunks": 0, "total_documents": 0}

    log("\n✂️ Chunking documents...")
    all_chunks = []
    chunk_counts_by_file = {}

    for doc in documents:
        if doc['content_length'] > 0:
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)
            chunk_counts_by_file[doc['file_name']] = {
                'count': len(chunks),
                'file_type': doc['file_type'],
            }
            log(f"  {doc['file_name']}: {len(chunks)} chunks")

    log(f"\nTotal chunks created: {len(all_chunks)}")

    log("\n🗑️ Clearing existing FAISS index...")
    faiss_store = FAISSVectorStore()
    faiss_store.clear_all()

    database.clear_all_documents()

    log("\n💾 Adding chunks to FAISS...")
    faiss_store.add_chunks(all_chunks)

    log("\n🗄️  Recording document metadata in SQLite...")
    for file_name, info in chunk_counts_by_file.items():
        title = file_name.rsplit('.', 1)[0]
        database.upsert_document(
            filename=file_name,
            title=title,
            file_type=info['file_type'],
            chunk_count=info['count'],
        )
        log(f"  Recorded: {file_name} ({info['count']} chunks)")

    stats = faiss_store.get_stats()
    log("\n" + "=" * 60)
    log("✅ FAISS Index Rebuilt Successfully!")
    log("=" * 60)
    log(f"Total chunks in FAISS: {stats['total_chunks']}")
    log(f"FAISS dimension: {stats['dimension']}")
    log(f"Index type: {stats['index_type']}")
    log(f"Documents recorded in SQLite: {len(database.list_documents())}")

    return {
        "total_chunks": stats['total_chunks'],
        "total_documents": len(database.list_documents()),
        "dimension": stats['dimension'],
        "index_type": stats['index_type'],
    }


def main():
    """CLI entry point — preserves the original `python rebuild_faiss.py` behaviour."""
    rebuild_index(docs_folder="../docs", verbose=True)


if __name__ == "__main__":
    main()