"""
Chunking Service
Splits documents into smaller pieces for better search
"""

from typing import List, Dict, Any

class DocumentChunker:
    """
    Takes large documents and splits them into smaller chunks
    Each chunk will be searched individually
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize the chunker
        
        Args:
            chunk_size: How many characters in each chunk
            chunk_overlap: How many characters overlap between chunks
                          (helps maintain context between chunks)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        print(f"Chunker initialized:")
        print(f"  - Chunk size: {chunk_size} characters")
        print(f"  - Overlap: {chunk_overlap} characters")
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks
        
        Args:
            text: The full text to split
            metadata: Information about the source (file name, page number, etc.)
            
        Returns:
            List of chunks with their metadata
        """
        if not text:
            return []
        
        chunks = []
        metadata = metadata or {}
        
        # Simple but effective chunking
        # We split by paragraphs first, then combine until we reach chunk_size
        
        # First, split by paragraphs
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_index'] = chunk_index
                chunk_metadata['chunk_length'] = len(current_chunk)
                
                chunks.append({
                    'text': current_chunk.strip(),
                    'metadata': chunk_metadata
                })
                
                chunk_index += 1
                
                # Start new chunk with overlap (take last part of previous chunk)
                if self.chunk_overlap > 0:
                    # Take the last chunk_overlap characters from previous chunk
                    overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                    current_chunk = overlap_text + "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk
        if current_chunk:
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_index'] = chunk_index
            chunk_metadata['chunk_length'] = len(current_chunk)
            
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': chunk_metadata
            })
        
        print(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a document that came from DocumentLoader
        
        Args:
            document: Dictionary with 'content' and other metadata
            
        Returns:
            List of chunks with metadata
        """
        content = document.get('content', '')
        metadata = {
            'source': document.get('file_name', 'unknown'),
            'file_path': document.get('file_path', ''),
            'file_type': document.get('file_type', '')
        }
        
        return self.chunk_text(content, metadata)


# Simple test
if __name__ == "__main__":
    chunker = DocumentChunker()
    
    test_text = """
    This is the first paragraph. It contains some important information about exams.
    
    This is the second paragraph. It talks about registration deadlines.
    
    This is the third paragraph. It explains the fee payment process.
    """
    
    chunks = chunker.chunk_text(test_text, {'source': 'test.txt'})
    
    print("\n--- Chunks Created ---")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i}:")
        print(f"  Length: {len(chunk['text'])} chars")
        print(f"  Preview: {chunk['text'][:100]}...")