"""
RAG Service with FAISS Vector Store
Implements Retrieval-Augmented Generation using FAISS as specified in proposal
"""

import ollama
import time
from typing import List, Dict, Any
from app.services.faiss_vector_store import FAISSVectorStore

# Simple cache for repeated questions
cache = {}
cache_ttl = 3600

# Conversation memory
conversation_memory = {}

class RAGServiceFAISS:
    """
    RAG Service using FAISS vector database and Ollama LLM
    """
    
    def __init__(self, model_name: str = "tinyllama"):
        self.model_name = model_name
        self.vector_store = FAISSVectorStore()  # FAISS instead of ChromaDB
        
        # Keywords for TU-K related questions
        self.tuk_keywords = [
            'tuk', 'technical university', 'kenya', 'exam', 'registration', 
            'fee', 'campus', 'library', 'student', 'course', 'department', 
            'lecture', 'academic', 'calendar', 'deadline', 'application', 
            'admission', 'semester', 'class', 'hostel', 'facility', 'graduation',
            'project', 'guideline', 'proposal', 'thesis', 'evaluation', 'presentation'
        ]
        
        # System prompt
        self.system_prompt = """You are TUK-ConvoSearch, an AI assistant for Technical University of Kenya.

CRITICAL RULES:
1. ONLY answer using information from the context below
2. If the context does NOT contain the answer, say: "I cannot find this information in the available TU-K documents."
3. NEVER use your general knowledge to answer
4. Use conversation history to understand follow-up questions

Conversation History:
{history}

Current Context from TU-K documents:
{context}

Question: {question}

Answer (using ONLY the context above):"""
        
        print(f"✓ RAG Service with FAISS initialized using {model_name}")
        stats = self.vector_store.get_stats()
        print(f"  FAISS index contains {stats['total_chunks']} chunks")
    
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
    
    def clear_history(self, session_id: str) -> bool:
        """Clear conversation history for a session"""
        if session_id in conversation_memory:
            del conversation_memory[session_id]
            return True
        return False
    
    def answer_question(self, question: str, session_id: str = "default", k: int = 5) -> Dict[str, Any]:
        """Answer a question using FAISS-based RAG"""
        print(f"\n🤔 Question: {question}")
        print(f"  Session: {session_id}")
        
        # Check if question is about TU-K
        question_lower = question.lower()
        is_tuk_related = any(kw in question_lower for kw in self.tuk_keywords)
        
        if not is_tuk_related and "what is your name" not in question_lower:
            return {
                "answer": "I'm TUK-ConvoSearch, your AI assistant for Technical University of Kenya. I can only answer questions about TU-K related topics like exams, registration, fees, campus facilities, and academic calendars. Please ask me something about TU-K!",
                "sources": [],
                "chunks_found": 0,
                "vector_db": "FAISS"
            }
        
        # Check cache
        cache_key = f"{session_id}_{question}_{self.model_name}"
        if cache_key in cache:
            cache_time, cache_result = cache[cache_key]
            if time.time() - cache_time < cache_ttl:
                print(f"  ⚡ Returning cached answer")
                return cache_result
        
        # Get conversation history
        history = self.get_conversation_history(session_id, limit=5)
        
        # Search using FAISS
        print(f"  🔍 Searching FAISS...")
        relevant_chunks = self.vector_store.search(question, k=k)
        
        if not relevant_chunks:
            return {
                "answer": "I cannot find this information in the available TU-K documents. Please contact the university directly for accurate information.",
                "sources": [],
                "chunks_found": 0,
                "vector_db": "FAISS"
            }
        
        # Build context and sources
        context_parts = []
        sources = []
        
        for chunk in relevant_chunks[:3]:
            source = chunk['metadata'].get('source', 'unknown')
            context_parts.append(chunk['text'][:500])
            sources.append({'source': source, 'distance': chunk.get('distance', 0)})
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
        print(f"  💭 Generating answer with {self.model_name}...")
        start_time = time.time()
        
        try:
            prompt = self.system_prompt.format(
                history=history,
                context=context,
                question=question
            )
            
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'user', 'content': prompt}
                ],
                options={
                    'num_predict': 250,
                    'temperature': 0.3,
                }
            )
            
            answer = response['message']['content']
            elapsed = time.time() - start_time
            print(f"  ✓ Answer generated in {elapsed:.1f} seconds")
            
        except Exception as e:
            answer = f"Sorry, I encountered an error: {e}"
            elapsed = 0
        
        result = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "chunks_found": len(relevant_chunks),
            "response_time": round(elapsed, 1),
            "vector_db": "FAISS",
            "total_chunks": self.vector_store.get_stats()['total_chunks']
        }
        
        # Cache result
        cache[cache_key] = (time.time(), result)
        
        # Add to conversation history
        self.add_to_history(session_id, "user", question)
        self.add_to_history(session_id, "assistant", answer)
        
        return result


# Simple test
if __name__ == "__main__":
    print("=" * 60)
    print("Testing RAG Service with FAISS")
    print("=" * 60)
    
    # Make sure Ollama is running
    print("\n⚠️ Make sure Ollama is running in another terminal!")
    
    rag = RAGServiceFAISS()
    result = rag.answer_question("What are the project guidelines?")
    
    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(result['answer'])
    print("\n" + "=" * 60)
    print("SOURCES:")
    print("=" * 60)
    for source in result['sources']:
        print(f"  - {source['source']} (distance: {source['distance']:.4f})")