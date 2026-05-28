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
        self.vector_store = FAISSVectorStore()
        
        # TU-K keywords and filter
        if "tell me a joke" in question_lower:
            return {
                "answer": "Why did the student bring a ladder to the library? Because they heard the books were on a higher level! 😄",
                "sources": [],
                "chunks_found": 0
            }

        if "what can you do" in question_lower or "your capabilities" in question_lower:
            return {
                "answer": "I can help you with:\n• Find exam dates and schedules\n• Answer questions about project guidelines\n• Locate campus facilities\n• Explain registration procedures\n• Provide fee information\n• And much more about TU-K!",
                "sources": [],
                "chunks_found": 0
        }
        self.tuk_keywords = [
            'tuk', 'technical university', 'kenya', 'exam', 'registration',
            'fee', 'campus', 'library', 'student', 'course', 'department',
            'lecture', 'academic', 'calendar', 'deadline', 'semester',
            'project', 'guideline', 'proposal', 'timetable', 'computing',
            'graduation', 'degree', 'diploma', 'upgrade', 'evaluation'
        ]
        
        # System prompt
        self.system_prompt = """You are TUK-ConvoSearch, an AI assistant for Technical University of Kenya.

CRITICAL RULES:
1. ONLY answer using information from the context below
2. If answer not in context say: "I cannot find this information in the available TU-K documents."
3. ALWAYS cite your sources - mention which document provided the information
4. Use proper spelling and grammar

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
        if session_id not in conversation_memory:
            return "No previous conversation."
        
        history = conversation_memory[session_id][-limit:]
        history_text = []
        for msg in history:
            role = "User" if msg['role'] == 'user' else "Assistant"
            history_text.append(f"{role}: {msg['content']}")
        
        return "\n".join(history_text)
    
    def add_to_history(self, session_id: str, role: str, content: str):
        if session_id not in conversation_memory:
            conversation_memory[session_id] = []
        
        conversation_memory[session_id].append({
            'role': role,
            'content': content,
            'timestamp': time.time()
        })
        
        if len(conversation_memory[session_id]) > 20:
            conversation_memory[session_id] = conversation_memory[session_id][-20:]
    
    def answer_question(self, question: str, session_id: str = "default", k: int = 5) -> Dict[str, Any]:
        print(f"\n🤔 Question: {question}")
        print(f"  Session: {session_id}")
        
        # Check if TU-K related
        question_lower = question.lower()
        is_tuk_related = any(kw in question_lower for kw in self.tuk_keywords)
        
        if not is_tuk_related and len(question_lower) > 5:
            return {
                "answer": "I'm TUK-ConvoSearch. I can only answer questions about TU-K related topics.",
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
        
        # Get history
        history = self.get_conversation_history(session_id, limit=5)
        
        # Search FAISS
        print(f"  🔍 Searching FAISS...")
        relevant_chunks = self.vector_store.search(question, k=k)
        
        if not relevant_chunks:
            return {
                "answer": "I cannot find this information in the available TU-K documents.",
                "sources": [],
                "chunks_found": 0,
                "vector_db": "FAISS"
            }
        
        # Build context and sources with quotes
        context_parts = []
        sources = []
        
        for chunk in relevant_chunks[:3]:
            source = chunk['metadata'].get('source', 'unknown')
            chunk_text = chunk['text']
            
            # Extract quote (first 250 characters)
            quote = chunk_text[:250].strip()
            if len(chunk_text) > 250:
                quote = quote + "..."
            
            context_parts.append(f"[Source: {source}]\n{chunk_text[:600]}")
            sources.append({
                'source': str(source),
                'quote': str(quote),
                'relevance_score': float(chunk.get('relevance_score', 0))
            })
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
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
                    'temperature': 0.2,
                }
            )
            
            answer = response['message']['content']
            elapsed = time.time() - start_time
            print(f"  ✓ Answer generated in {elapsed:.1f} seconds")
            
        except Exception as e:
            answer = f"Error generating answer: {e}"
            elapsed = 0
        
        result = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "chunks_found": len(relevant_chunks),
            "response_time": float(elapsed),
            "vector_db": "FAISS",
            "total_chunks": self.vector_store.get_stats()['total_chunks']
        }
        
        # Cache result
        cache[cache_key] = (time.time(), result)
        
        # Add to history
        self.add_to_history(session_id, "user", question)
        self.add_to_history(session_id, "assistant", answer)
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'vector_db': self.vector_store.get_stats(),
            'model': self.model_name,
            'cache_size': len(cache),
            'active_sessions': len(conversation_memory)
        }
    
    def clear_cache(self):
        global cache
        cache = {}
        print("✓ Cache cleared")
    
    def clear_history(self, session_id: str = None):
        global conversation_memory
        if session_id:
            conversation_memory.pop(session_id, None)
            print(f"✓ Cleared history for session: {session_id}")
        else:
            conversation_memory = {}
            print("✓ Cleared all conversation history")

asynch def stream_answer(self, question: str, session_id:str = "default", k: int = 5):
    """Stream answer word by word"""

    # Search for relevant chunks (same as before)
    relevant_chunks = self.vector_store.search(question, k=k)

    if not relevant_chunks:
        yield "I cannot find this information in the available TU-K documents. "
        return

    # Build context
    context_parts = []
    sources = []
    for chunks in relevant_chunks[:3]:
        source = chunk['metadata'].get('source', 'unkown')
        context_parts.append(chunk['text'][:600])
        sources.append({'source': source})

    context = "\n\n".join(context_parts)

    # Stream from Ollama
    try:
        stream = ollama.chat(
            model = self.model_name,
            messages=[
                {'role': 'user'. 'content': self.system_prompt.format(
                    context=context,
                    history="",
                    question=question
                )}
            ],
            stream=True # This enables streaming!
        )

        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']

    except Exception as e:
        yield f"Error: {e}"   
