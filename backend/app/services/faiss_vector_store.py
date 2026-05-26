"""
FAISS Vector Store Service - Fixed for JSON serialization
Location: backend/app/services/faiss_vector_store.py
"""

import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

class FAISSVectorStore:
    def __init__(self, dimension: int = 384, persist_directory: str = "./faiss_index"):
        self.dimension = dimension
        self.persist_directory = persist_directory
        self.index_file = os.path.join(persist_directory, "faiss_index.bin")
        self.metadata_file = os.path.join(persist_directory, "metadata.pkl")
        
        os.makedirs(persist_directory, exist_ok=True)
        
        print("  Loading embedding model for FAISS...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
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
        if not chunks:
            return 0
        
        print(f"Adding {len(chunks)} chunks to FAISS...")
        
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedder.encode(texts)
        
        self.index.add(np.array(embeddings).astype('float32'))
        
        for i, chunk in enumerate(chunks):
            self.metadata.append({
                'id': len(self.metadata),
                'text': chunk['text'],
                'metadata': chunk['metadata']
            })
        
        self._save()
        
        print(f"✓ Added {len(chunks)} chunks to FAISS")
        print(f"  Total chunks in FAISS: {self.index.ntotal}")
        
        return len(chunks)
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        print(f"  FAISS searching for: '{query}'")
        
        query_embedding = self.embedder.encode([query])
        
        k = min(k, self.index.ntotal)
        if k == 0:
            return []
        
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), 
            k
        )
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):
                chunk_data = self.metadata[idx]
                # Convert numpy types to Python native types for JSON serialization
                distance = float(distances[0][i])
                results.append({
                    'text': str(chunk_data['text']),
                    'metadata': {
                        'source': str(chunk_data['metadata'].get('source', 'unknown')),
                        'file_path': str(chunk_data['metadata'].get('file_path', '')),
                        'chunk_index': int(chunk_data['metadata'].get('chunk_index', 0))
                    },
                    'distance': distance,
                    'relevance_score': float(1 / (1 + distance))
                })
        
        print(f"  Found {len(results)} relevant chunks")
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_chunks': int(self.index.ntotal),
            'dimension': int(self.dimension),
            'persist_directory': str(self.persist_directory),
            'index_type': 'IndexFlatL2'
        }
    
    def clear_all(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        self._save()
        print("✓ FAISS index cleared")
    
    def _save(self):
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)