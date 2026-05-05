"""
FAISS Vector Store Service
Implements FAISS-based vector storage as specified in the proposal
Location: backend/app/services/faiss_vector_store.py
"""

import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

class FAISSVectorStore:
    """
    FAISS-based vector database for storing and searching document chunks
    """
    
    def __init__(self, dimension: int = 384, persist_directory: str = "./faiss_index"):
        """
        Initialize FAISS index
        
        Args:
            dimension: Embedding dimension (384 for all-MiniLM-L6-v2)
            persist_directory: Where to save the FAISS index
        """
        self.dimension = dimension
        self.persist_directory = persist_directory
        self.index_file = os.path.join(persist_directory, "faiss_index.bin")
        self.metadata_file = os.path.join(persist_directory, "metadata.pkl")
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Load embedding model
        print("  Loading embedding model for FAISS...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load or create FAISS index
        if os.path.exists(self.index_file):
            print(f"  Loading existing FAISS index from {self.index_file}")
            self.index = faiss.read_index(self.index_file)
            with open(self.metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"  Loaded {len(self.metadata)} chunks")
        else:
            print("  Creating new FAISS index (Flat L2)")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = []
            print("  New index created")
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Add chunks to FAISS index
        
        Args:
            chunks: List of dictionaries with 'text' and 'metadata'
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        print(f"Adding {len(chunks)} chunks to FAISS...")
        
        # Extract texts and generate embeddings
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedder.encode(texts)
        
        # Add to FAISS index
        self.index.add(np.array(embeddings).astype('float32'))
        
        # Store metadata
        for i, chunk in enumerate(chunks):
            self.metadata.append({
                'id': len(self.metadata),
                'text': chunk['text'],
                'metadata': chunk['metadata']
            })
        
        # Save to disk
        self._save()
        
        print(f"✓ Added {len(chunks)} chunks to FAISS")
        print(f"  Total chunks in FAISS: {self.index.ntotal}")
        
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
        print(f"  FAISS searching for: '{query}'")
        
        # Convert query to embedding
        query_embedding = self.embedder.encode([query])
        
        # Search FAISS
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), 
            min(k, self.index.ntotal)
        )
        
        # Format results with proper JSON-serializable types
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):
                chunk_data = self.metadata[idx]
                # Convert numpy types to Python native types
                distance = float(distances[0][i])  # Convert to Python float
                results.append({
                    'text': chunk_data['text'],
                    'metadata': {
                        'source': chunk_data['metadata'].get('source', 'unknown'),
                        'file_path': chunk_data['metadata'].get('file_path', ''),
                        'chunk_index': int(chunk_data['metadata'].get('chunk_index', 0))  # Convert to int
                    },
                    'distance': distance,
                    'relevance_score': float(1 / (1 + distance))  # Convert to Python float
                })
        
        print(f"  Found {len(results)} relevant chunks")
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the FAISS index"""
        return {
            'total_chunks': self.index.ntotal,
            'dimension': self.dimension,
            'persist_directory': self.persist_directory,
            'index_type': 'IndexFlatL2'
        }
    
    def clear_all(self):
        """Delete all documents from FAISS index"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        self._save()
        print("✓ FAISS index cleared")
    
    def _save(self):
        """Save FAISS index and metadata to disk"""
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def get_chunk_by_id(self, chunk_id: int) -> Dict[str, Any]:
        """Retrieve a specific chunk by its ID"""
        if 0 <= chunk_id < len(self.metadata):
            return self.metadata[chunk_id]
        return None