"""
Rebuild FAISS index from all documents in docs folder
Location: backend/rebuild_faiss.py

Now also syncs document-level metadata into SQLite (documents table)
so the React admin panel (Phase 3) has something durable to read from.
FAISS remains the only place chunk-level vectors live; SQLite tracks
which files were ingested, when, and how many chunks each produced.
"""

from app.services.document_loader import DocumentLoader
from app.services.chunking import DocumentChunker
from app.services.faiss_vector_store import FAISSVectorStore
from app.services import database


def main():
    print("=" * 60)
    print("Rebuilding FAISS Index from All Documents")
    print("=" * 60)

    # Step 0: Make sure the SQLite schema exists
    database.init_db()

    # Step 1: Load documents
    print("\n📁 Loading documents from docs folder...")
    loader = DocumentLoader()
    chunker = DocumentChunker()

    documents = loader.load_documents_from_folder("../docs")
    print(f"Loaded {len(documents)} documents")

    if not documents:
        print("No documents found!")
        return

    # Step 2: Chunk all documents
    print("\n✂️ Chunking documents...")
    all_chunks = []
    chunk_counts_by_file = {}

    for doc in documents:
        if doc['content_length'] > 0:  # Skip empty documents
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)
            chunk_counts_by_file[doc['file_name']] = {
                'count': len(chunks),
                'file_type': doc['file_type'],
            }
            print(f"  {doc['file_name']}: {len(chunks)} chunks")

    print(f"\nTotal chunks created: {len(all_chunks)}")

    # Step 3: Clear existing FAISS index
    print("\n🗑️ Clearing existing FAISS index...")
    faiss_store = FAISSVectorStore()
    faiss_store.clear_all()

    # Step 3b: Clear existing documents table so it stays in sync with
    # the freshly rebuilt FAISS index (this script always does a full rebuild)
    database.clear_all_documents()

    # Step 4: Add chunks to FAISS
    print("\n💾 Adding chunks to FAISS...")
    faiss_store.add_chunks(all_chunks)

    # Step 4b: Record each document's metadata in SQLite
    print("\n🗄️  Recording document metadata in SQLite...")
    for file_name, info in chunk_counts_by_file.items():
        # Use the filename (without extension) as a human-readable title;
        # admins can rename this later via the admin panel if needed.
        title = file_name.rsplit('.', 1)[0]
        database.upsert_document(
            filename=file_name,
            title=title,
            file_type=info['file_type'],
            chunk_count=info['count'],
        )
        print(f"  Recorded: {file_name} ({info['count']} chunks)")

    # Step 5: Verify
    stats = faiss_store.get_stats()
    print("\n" + "=" * 60)
    print("✅ FAISS Index Rebuilt Successfully!")
    print("=" * 60)
    print(f"Total chunks in FAISS: {stats['total_chunks']}")
    print(f"FAISS dimension: {stats['dimension']}")
    print(f"Index type: {stats['index_type']}")
    print(f"Documents recorded in SQLite: {len(database.list_documents())}")


if __name__ == "__main__":
    main()