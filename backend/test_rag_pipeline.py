"""
Test script for the complete RAG pipeline
This will:
1. Load documents from the docs folder
2. Chunk them into smaller pieces
3. Create embeddings and store in vector database
4. Search for answers to questions
"""

from app.services.document_loader import DocumentLoader
from app.services.chunking import DocumentChunker
from app.services.vector_store import VectorStore
import os

def main():
    print("=" * 60)
    print("TUK-ConvoSearch - RAG Pipeline Test")
    print("=" * 60)
    
    # Step 1: Initialize our services
    print("\n📦 Initializing services...")
    loader = DocumentLoader()
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
    vector_store = VectorStore(persist_directory="./chroma_db")
    
    # Step 2: Load documents from docs folder
    docs_folder = "..\\docs"
    print(f"\n📁 Loading documents from: {docs_folder}")
    
    if not os.path.exists(docs_folder):
        print(f"  Docs folder not found!")
        return
    
    documents = loader.load_documents_from_folder(docs_folder)
    print(f"  Loaded {len(documents)} documents")
    
    if not documents:
        print("  No documents found. Please add files to the docs folder")
        return
    
    # Step 3: Chunk each document
    print("\n✂️  Chunking documents...")
    all_chunks = []
    
    for doc in documents:
        print(f"  Chunking: {doc['file_name']} ({doc['content_length']} chars)")
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"    → Created {len(chunks)} chunks")
    
    print(f"\n  Total chunks created: {len(all_chunks)}")
    
    # Step 4: Add chunks to vector database
    print("\n💾 Adding chunks to vector database...")
    vector_store.add_chunks(all_chunks)
    
    # Step 5: Show stats
    stats = vector_store.get_stats()
    print(f"\n📊 Vector Store Statistics:")
    print(f"  Total chunks stored: {stats['total_chunks']}")
    
    # Step 6: Test searches
    print("\n" + "=" * 60)
    print("🔍 Testing Search Functionality")
    print("=" * 60)
    
    test_questions = [
        "When do exams start?",
        "What are the registration dates?",
        "When are fees due?",
        "Where is the university located?"
    ]
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        print("-" * 40)
        
        results = vector_store.search(question, k=3)
        
        if results:
            print(f"  Found {len(results)} relevant chunks:")
            for i, result in enumerate(results):
                print(f"\n  Result {i+1}:")
                print(f"    Source: {result['metadata'].get('source', 'unknown')}")
                print(f"    Text: {result['text'][:200]}...")
                if result['distance']:
                    print(f"    Relevance: {1 - result['distance']:.2%}")
        else:
            print("  No relevant chunks found")
    
    print("\n" + "=" * 60)
    print("✅ Test Complete!")
    print("=" * 60)
    print("\n💡 Try asking your own questions by modifying the test_questions list")

if __name__ == "__main__":
    main()