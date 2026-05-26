"""
Test FAISS Vector Store
"""

from app.services.faiss_vector_store import FAISSVectorStore
from app.services.document_loader import DocumentLoader
from app.services.chunking import DocumentChunker

def main():
    print("=" * 60)
    print("Testing FAISS Vector Store")
    print("=" * 60)
    
    # Step 1: Load documents
    print("\n📁 Loading documents...")
    loader = DocumentLoader()
    chunker = DocumentChunker()
    
    docs_folder = "../docs"
    documents = loader.load_documents_from_folder(docs_folder)
    print(f"Loaded {len(documents)} documents")
    
    if not documents:
        print("No documents found. Add files to docs folder.")
        return
    
    # Step 2: Chunk documents
    print("\n✂️ Chunking documents...")
    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)
    print(f"Created {len(all_chunks)} chunks")
    
    # Step 3: Add to FAISS
    print("\n💾 Adding chunks to FAISS...")
    faiss_store = FAISSVectorStore()
    faiss_store.add_chunks(all_chunks)
    
    # Step 4: Show stats
    stats = faiss_store.get_stats()
    print(f"\n📊 FAISS Statistics:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Dimension: {stats['dimension']}")
    
    # Step 5: Test search
    print("\n🔍 Testing search...")
    test_queries = [
        "When do exams start?",
        "What are the project guidelines?",
        "Where is the university located?"
    ]
    
    for query in test_queries:
        print(f"\n❓ Query: {query}")
        results = faiss_store.search(query, k=3)
        for i, result in enumerate(results):
            print(f"  Result {i+1}: {result['metadata'].get('source', 'unknown')[:50]}")
            print(f"    Distance: {result['distance']:.4f}")
    
    print("\n✅ FAISS test complete!")

if __name__ == "__main__":
    main()