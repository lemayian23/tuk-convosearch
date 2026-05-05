"""
Test script for Proposal-Compliant RAG Service
Location: backend/test_proposal.py
"""

from app.services.rag_service_proposal import RAGServiceProposal

def main():
    print("=" * 60)
    print("Testing Proposal-Compliant RAG Service")
    print("=" * 60)
    
    # Initialize service
    rag = RAGServiceProposal()
    
    # Test questions
    test_questions = [
        "What are the project guidelines?",
        "When do exams start?",
        "Where is the university located?",
        "What is the exam timetable for BTECH groups?",
        "How do I upgrade from diploma to degree?"
    ]
    
    for question in test_questions:
        print(f"\n--- Question: {question} ---")
        result = rag.answer_question(question)
        print(f"Answer: {result['answer'][:300]}...")
        print(f"Sources: {[s['source'] for s in result['sources']]}")
        print(f"Response time: {result['response_time']}s")
        print(f"Vector DB: {result['vector_db']}")
        print("-" * 40)
    
    # Show statistics
    print("\n" + "=" * 60)
    print("System Statistics:")
    print("=" * 60)
    stats = rag.get_stats()
    print(f"Total chunks in FAISS: {stats['vector_db']['total_chunks']}")
    print(f"FAISS dimension: {stats['vector_db']['dimension']}")
    print(f"Cache size: {stats['cache_size']}")
    print(f"Active sessions: {stats['active_sessions']}")

if __name__ == "__main__":
    main()