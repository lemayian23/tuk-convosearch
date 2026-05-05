"""
RAG Service - Fully compliant with proposal specifications
Uses FAISS vector database as specified
Location: backend/app/services/rag_service_proposal.py
"""

import ollama
import time
from typing import List, Dict, Any
from app.services.faiss_vector_store import FAISSVectorStore

# Cache for repeated questions
cache = {}
cache_ttl = 3600

# Conversation memory
conversation_memory = {}

class RAGServiceProposal:
    """
    RAG Service implementing all proposal specifications:
    - FAISS vector database
    - Transformer-based embeddings
    - LLM response generation with source citations
    """
    
    def __init__(self, model_name: str = "llama3.2:1b"):
        self.model_name = model_name
        self.vector_store = FAISSVectorStore()  # FAISS as specified
        
        # TU-K keywords for topic filtering
        self.tuk_keywords = [
            'tuk', 'technical university', 'kenya', 'exam', 'registration',
            'fee', 'campus', 'library', 'student', 'course', 'department',
            'lecture', 'academic', 'calendar', 'deadline', 'semester',
            'project', 'guideline', 'proposal', 'timetable', 'computing',
            'graduation', 'degree', 'diploma', 'upgrade', 'evaluation',
            'presentation', 'thesis', 'research', 'btech', 'exam timetable'
        ]
        
        # System prompt with citation requirement
        self.system_prompt = """You are TUK-ConvoSearch, an AI assistant for Technical University of Kenya.

CRITICAL RULES - PROPOSAL REQUIREMENTS:
1. ONLY answer using information from the context below
2. If answer not in context say: "I cannot find this information in the available TU-K documents."
3. ALWAYS cite your sources - mention which document provided the information
4. Use proper spelling and grammar
5. Keep answers concise and helpful

CONTEXT (from TU-K documents):
{context}

CONVERSATION HISTORY:
{history}

QUESTION: {question}

ANSWER (with source citations):"""
        
        print(f"✓ Proposal-Compliant RAG Service Initialized")
        print(f"  LLM Model: {self.model_name}")
        stats = self.vector_store.get_stats()
        print(f"  FAISS Vector DB: {stats['total_chunks']} chunks, {stats['dimension']}-dim vectors")
    
    def get_conversation_history(self, session_id: str, limit: int = 5) -> str:
        """Get recent conversation history for a session"""
        if session_id not in conversation_memory:
            return "No previous conversation."
        
        history = conversation_memory[session_id][-limit:]
        history_text = []
        for msg in history:
            role = "User" if msg['role'] == 'user' else "Assistant"
            history_text.append(f"{role}: {msg['content']}")
        
        return "\n".join(history_text)
    
    def add_to_history(self, session_id: str, role: str, content: str):
        """Add a message to conversation history"""
        if session_id not in conversation_memory:
            conversation_memory[session_id] = []
        
        conversation_memory[session_id].append({
            'role': role,
            'content': content,
            'timestamp': time.time()
        })
        
        # Keep only last 20 messages per session
        if len(conversation_memory[session_id]) > 20:
            conversation_memory[session_id] = conversation_memory[session_id][-20:]
    
    def answer_question(self, question: str, session_id: str = "default", k: int = 5) -> Dict[str, Any]:
        """
        Answer a question using FAISS-based RAG
        
        Args:
            question: User's question
            session_id: Session identifier for conversation memory
            k: Number of chunks to retrieve
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        print(f"\n🤔 Question: {question}")
        print(f"  Session: {session_id}")
        
        # Check if question is TU-K related
        question_lower = question.lower()
        is_tuk_related = any(kw in question_lower for kw in self.tuk_keywords)
        
        if not is_tuk_related and len(question_lower) > 5:
            return {
                "answer": "I'm TUK-ConvoSearch, your AI assistant for Technical University of Kenya. I can only answer questions about TU-K related topics.",
                "sources": [],
                "chunks_found": 0,
                "vector_db": "FAISS"
            }
        
        # Check cache for identical questions
        cache_key = f"{session_id}_{question}_{self.model_name}"
        if cache_key in cache:
            cache_time, cache_result = cache[cache_key]
            if time.time() - cache_time < cache_ttl:
                print(f"  ⚡ Returning cached answer")
                return cache_result
        
        # Get conversation history for context
        history = self.get_conversation_history(session_id, limit=5)
        
        # Search FAISS vector database
        print(f"  🔍 Searching FAISS vector database...")
        relevant_chunks = self.vector_store.search(question, k=k)
        
        if not relevant_chunks:
            return {
                "answer": "I cannot find this information in the available TU-K documents. Please contact the university directly or add more documents.",
                "sources": [],
                "chunks_found": 0,
                "vector_db": "FAISS"
            }
        
              # Build context from retrieved chunks with JSON-safe types
        context_parts = []
        sources = []
        
        for chunk in relevant_chunks[:3]:
            source = chunk['metadata'].get('source', 'unknown')
            context_parts.append(f"[Source: {source}]\n{chunk['text'][:600]}")
            sources.append({
                'source': str(source),  # Ensure string type
                'relevance_score': float(chunk.get('relevance_score', 0))  # Convert to float
            })
        
        context = "\n\n".join(context_parts)
        
        # Generate response using LLM
        print(f"  💭 Generating answer with {self.model_name}...")
        start_time = time.time()
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'user', 'content': self.system_prompt.format(
                        context=context,
                        history=history,
                        question=question
                    )}
                ],
                options={
                    'num_predict': 300,
                    'temperature': 0.2,  # Lower temperature = more factual
                }
            )
            
            answer = response['message']['content']
            elapsed = time.time() - start_time
            print(f"  ✓ Answer generated in {elapsed:.1f} seconds")
            
        except Exception as e:
            answer = f"Error generating answer: {e}"
            elapsed = 0
        
        # Prepare result
        result = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "chunks_found": len(relevant_chunks),
            "response_time": round(elapsed, 1),
            "vector_db": "FAISS",
            "total_chunks": self.vector_store.get_stats()['total_chunks']
        }
        
        # Cache the result
        cache[cache_key] = (time.time(), result)
        
        # Store in conversation history
        self.add_to_history(session_id, "user", question)
        self.add_to_history(session_id, "assistant", answer)
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            'vector_db': self.vector_store.get_stats(),
            'model': self.model_name,
            'cache_size': len(cache),
            'active_sessions': len(conversation_memory)
        }
    
    def clear_cache(self):
        """Clear the response cache"""
        global cache
        cache = {}
        print("✓ Cache cleared")
    
    def clear_history(self, session_id: str = None):
        """Clear conversation history for a session or all sessions"""
        global conversation_memory
        if session_id:
            conversation_memory.pop(session_id, None)
            print(f"✓ Cleared history for session: {session_id}")
        else:
            conversation_memory = {}
            print("✓ Cleared all conversation history")


# Test the service
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Proposal-Compliant RAG Service")
    print("=" * 60)
    
    rag = RAGServiceProposal()
    
    # Test questions
    test_questions = [
        "What are the project guidelines?",
        "Where is the university located?",
        "What is the exam timetable for BTECH groups?"
    ]
    
    for question in test_questions:
        result = rag.answer_question(question)
        print(f"\n--- Question: {question} ---")
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Sources: {[s['source'] for s in result['sources']]}")
        print(f"Vector DB: {result['vector_db']}")
        print(f"Response time: {result['response_time']}s")
    
    print("\n" + "=" * 60)
    print("Stats:", rag.get_stats())