"""
Interactive Question Asking
Type your questions and get answers from your RAG system!
"""

from app.services.vector_store import VectorStore

def main():
    print("=" * 60)
    print("TUK-ConvoSearch - Ask Me Anything!")
    print("=" * 60)
    print("\nType your questions about TU-K")
    print("Type 'quit' to exit\n")
    
    # Initialize the vector store
    vector_store = VectorStore()
    
    stats = vector_store.get_stats()
    print(f"📚 Loaded {stats['total_chunks']} document chunks")
    print()
    
    while True:
        # Get question from user
        question = input("❓ Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not question:
            continue
        
        print("\n🔍 Searching...")
        
        # Search for relevant chunks
        results = vector_store.search(question, k=3)
        
        if results:
            print(f"\n📖 Found {len(results)} relevant sections:\n")
            
            for i, result in enumerate(results):
                print(f"--- Source {i+1}: {result['metadata'].get('source', 'unknown')} ---")
                print(result['text'][:500])
                print()
        else:
            print("\n❌ No relevant information found.")
            print("   Try rephrasing your question or add more documents.\n")

if __name__ == "__main__":
    main()