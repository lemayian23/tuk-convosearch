"""
Rebuild FAISS index from all documents in docs folder
Location: backend/rebuild_faiss.py
"""

from app.services.document_loader import DocumentLoader
from app.services.chunking import DocumentChunker
from app.services.faiss_vector_store import FAISSVectorStore

def main():
    print("=" * 60)
    print("Rebuilding FAISS Index from All Documents")
    print("=" * 60)
    
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
    for doc in documents:
        if doc['content_length'] > 0:  # Skip empty documents
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)
            print(f"  {doc['file_name']}: {len(chunks)} chunks")
    
    print(f"\nTotal chunks created: {len(all_chunks)}")
    
    # Step 3: Clear existing FAISS index
    print("\n🗑️ Clearing existing FAISS index...")
    faiss_store = FAISSVectorStore()
    faiss_store.clear_all()
    
    # Step 4: Add chunks to FAISS
    print("\n💾 Adding chunks to FAISS...")
    faiss_store.add_chunks(all_chunks)
    
    # Step 5: Verify
    stats = faiss_store.get_stats()
    print("\n" + "=" * 60)
    print("✅ FAISS Index Rebuilt Successfully!")
    print("=" * 60)
    print(f"Total chunks in FAISS: {stats['total_chunks']}")
    print(f"FAISS dimension: {stats['dimension']}")
    print(f"Index type: {stats['index_type']}")

if __name__ == "__main__":
    main()