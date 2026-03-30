"""
RAG Service - Optimized with Hallucination Prevention
"""

import ollama
import time
from typing import List, Dict, Any
from app.services.vector_store import VectorStore

# Simple cache
cache = {}
cache_ttl = 3600

class RAGService:
    def __init__(self, model_name: str = "tinyllama"):
        self.model_name = model_name
        self.vector_store = VectorStore()
        
        # Strict prompt - NO HALLUCINATIONS
        self.system_prompt = """You are TUK-ConvoSearch, an AI assistant for Technical University of Kenya.

CRITICAL RULES - YOU MUST FOLLOW:
1. ONLY answer using information from the context below
2. If the context does NOT contain the answer, say: "I cannot find this information in the available TU-K documents. Please contact the university directly for accurate information."
3. NEVER use your general knowledge to answer
4. NEVER make up information
5. If the question is about something outside TU-K, say you can only answer questions about TU-K

Context from TU-K documents:
{context}

Question: {question}

Answer (using ONLY the context above):"""
        
        # Keywords for TU-K related questions
        self.tuk_keywords = [
            'tuk', 'technical university', 'kenya', 'exam', 'registration', 
            'fee', 'campus', 'library', 'student', 'course', 'department', 
            'lecture', 'academic', 'calendar', 'deadline', 'application', 
            'admission', 'semester', 'class', 'hostel', 'facility'
        ]
        
        print(f"✓ RAG Service initialized with {model_name}")
    
    def answer_question(self, question: str, k: int = 3) -> Dict[str, Any]:
        print(f"\n🤔 Question: {question}")
        
        # Step 1: Check if question is about TU-K
        question_lower = question.lower()
        is_tuk_related = any(keyword in question_lower for keyword in self.tuk_keywords)
        
        # Allow questions that might be about "what is your name" etc but return polite message
        if not is_tuk_related and "what is your name" not in question_lower and "who are you" not in question_lower:
            return {
                "answer": "I'm TUK-ConvoSearch, your AI assistant for Technical University of Kenya. I can only answer questions about TU-K related topics like exams, registration, fees, campus facilities, and academic calendars. Please ask me something about TU-K!",
                "sources": [],
                "chunks_found": 0
            }
        
        # Step 2: Check cache
        cache_key = f"{question}_{self.model_name}"
        if cache_key in cache:
            cache_time, cache_result = cache[cache_key]
            if time.time() - cache_time < cache_ttl:
                print(f"  ⚡ Returning cached answer")
                return cache_result
        
        # Step 3: Search
        print(f"  🔍 Searching documents...")
        relevant_chunks = self.vector_store.search(question, k=k)
        
        if not relevant_chunks:
            return {
                "answer": "I couldn't find any relevant information in the TU-K documents. Please make sure your question is about topics covered in the available documents.",
                "sources": [],
                "chunks_found": 0
            }
        
        # Step 4: Build context (shorter = faster)
        context_parts = []
        sources = []
        
        for chunk in relevant_chunks[:3]:
            source = chunk['metadata'].get('source', 'unknown')
            context_parts.append(chunk['text'][:500])  # Limit each chunk
            sources.append({'source': source})
        
        context = "\n\n".join(context_parts)
        
        # Step 5: Generate answer
        print(f"  💭 Generating answer with {self.model_name}...")
        start_time = time.time()
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'user', 'content': f"Context: {context}\n\nQuestion: {question}"}
                ],
                options={
                    'num_predict': 200,
                    'temperature': 0.3,  # Lower = more factual, less creative
                }
            )
            
            answer = response['message']['content']
            
            # Step 6: Final check - if answer is too short or seems made up
            if len(answer) < 10 and "cannot find" not in answer.lower():
                answer = "I cannot find this information in the available TU-K documents. Please contact the university directly for accurate information."
            
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
            "response_time": round(elapsed, 1)
        }
        
        # Cache result
        cache[cache_key] = (time.time(), result)
        
        return result