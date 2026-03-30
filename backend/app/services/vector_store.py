"""
Vector Store Service
Converts text to numbers (embeddings) and stores them for searching
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import time

class VectorStore:
    """
    Manages the vector database where we store document chunks
    Converts text to embeddings and searches for similar content
    """
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize the vector store
        
        Args:
            persist_directory: Where to save the database on disk
        """
        self.persist_directory = persist_directory
        
        print("Initializing Vector Store...")
        
        # Create the embedding model (converts text to numbers)
        # 'all-MiniLM-L6-v2' is a small, fast, good quality model
        print("  Loading embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create or connect to ChromaDB
        print(f"  Connecting to database at: {persist_directory}")
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create our collection (like a table in a database)
        self.collection = self.client.get_or_create_collection(
            name="tuk_documents",
            metadata={"description": "TU-K documents for RAG system"}
        )
        
        print(f"✓ Vector Store initialized")
        print(f"  Collection has {self.collection.count()} documents")
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Add chunks to the vector database
        
        Args:
            chunks: List of dictionaries with 'text' and 'metadata'
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            print("No chunks to add")
            return 0
        
        print(f"Adding {len(chunks)} chunks to vector store...")
        
        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            # Create a unique ID
            source = chunk['metadata'].get('source', 'unknown')
            chunk_id = f"{source}_{i}_{int(time.time())}"
            
            ids.append(chunk_id)
            documents.append(chunk['text'])
            metadatas.append(chunk['metadata'])
            
            # Generate embedding for this chunk
            embedding = self.embedder.encode(chunk['text'])
            embeddings.append(embedding.tolist())
        
        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"✓ Added {len(chunks)} chunks")
        print(f"  Total documents in store: {self.collection.count()}")
        
        return len(chunks)
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for chunks similar to the query
        
        Args:
            query: The user's question
            k: How many results to return
            
        Returns:
            List of relevant chunks with their metadata
        """
        print(f"Searching for: '{query}'")
        
        # Convert query to embedding
        query_embedding = self.embedder.encode([query])
        
        # Search the database
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k
        )
        
        # Format the results
        formatted_results = []
        
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None
                })
        
        print(f"  Found {len(formatted_results)} relevant chunks")
        return formatted_results
    
    def clear_all(self):
        """Delete all documents from the database"""
        count = self.collection.count()
        if count > 0:
            # Get all IDs and delete them
            all_ids = self.collection.get()['ids']
            self.collection.delete(ids=all_ids)
            print(f"✓ Cleared {count} documents from vector store")
        else:
            print("Database already empty")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return {
            'total_chunks': self.collection.count(),
            'persist_directory': self.persist_directory,
            'collection_name': self.collection.name
        }


# Simple test
if __name__ == "__main__":
    # Test the vector store
    store = VectorStore()
    
    # Add some test chunks
    test_chunks = [
        {
            'text': 'Exams start on April 20, 2026',
            'metadata': {'source': 'calendar.txt', 'topic': 'exams'}
        },
        {
            'text': 'Registration opens January 5, 2026',
            'metadata': {'source': 'calendar.txt', 'topic': 'registration'}
        },
        {
            'text': 'Fees must be paid by January 9, 2026',
            'metadata': {'source': 'fees.txt', 'topic': 'payments'}
        }
    ]
    
    store.add_chunks(test_chunks)
    
    # Test search
    results = store.search("When do exams start?")
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Text: {result['text']}")
        print(f"  Source: {result['metadata'].get('source')}")