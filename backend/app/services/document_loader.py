"""
Document Loader Service
Handles loading text from PDF, DOCX, and TXT files
"""

import os
from typing import List, Dict, Any
import PyPDF2
import pdfplumber
from docx import Document

class DocumentLoader:
    """
    A service class to load and extract text from various document formats
    """
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.txt']
    
    def load_document(self, file_path: str) -> Dict[str, Any]:
        """
        Load a single document and extract its text
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)
        
        print(f"Loading: {file_name}")
        
        if file_ext == '.pdf':
            text = self._load_pdf(file_path)
        elif file_ext == '.docx':
            text = self._load_docx(file_path)
        elif file_ext == '.txt':
            text = self._load_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        return {
            "file_name": file_name,
            "file_path": file_path,
            "file_type": file_ext,
            "content": text,
            "content_length": len(text)
        }
    
    def load_documents_from_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """
        Load all supported documents from a folder
        """
        documents = []
        
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        for file_name in os.listdir(folder_path):
            file_ext = os.path.splitext(file_name)[1].lower()
            if file_ext in self.supported_formats:
                file_path = os.path.join(folder_path, file_name)
                try:
                    doc = self.load_document(file_path)
                    documents.append(doc)
                    print(f"✓ Loaded: {file_name} ({doc['content_length']} chars)")
                except Exception as e:
                    print(f"✗ Failed to load {file_name}: {str(e)}")
        
        return documents
    
    def _load_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text_parts = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"--- Page {page_num} ---\n{page_text}")
            print(f"  Used pdfplumber for {os.path.basename(file_path)}")
        except Exception as e:
            print(f"  pdfplumber failed: {e}")
            # Try PyPDF2 as fallback
            try:
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page_num, page in enumerate(reader.pages, 1):
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(f"--- Page {page_num} ---\n{page_text}")
                print(f"  Used PyPDF2 as fallback")
            except Exception as e:
                print(f"  Both PDF methods failed: {e}")
                return ""
        
        return "\n\n".join(text_parts)
    
    def _load_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        text_parts = []
        
        try:
            doc = Document(file_path)
            for para in doc.paragraphs:
                if para.text.strip():
                    text_parts.append(para.text)
            
            print(f"  Extracted {len(text_parts)} paragraphs")
            return "\n\n".join(text_parts)
        except Exception as e:
            print(f"  Failed to load DOCX: {e}")
            return ""
    
    def _load_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            print(f"  Loaded TXT with utf-8 encoding")
            return content
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    content = file.read()
                print(f"  Loaded TXT with latin-1 encoding")
                return content
            except Exception as e:
                print(f"  Failed to load TXT: {e}")
                return ""