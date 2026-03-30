"""
Test script for Document Loader
"""

from app.services.document_loader import DocumentLoader
import os

def main():
    print("=" * 50)
    print("TUK-ConvoSearch - Document Loader Test")
    print("=" * 50)
    
    loader = DocumentLoader()
    print(f"\n✓ Document Loader initialized")
    print(f"  Supported formats: {loader.supported_formats}")
    
    # Go up one level to docs folder
    docs_folder = "..\\docs"
    
    if os.path.exists(docs_folder):
        print(f"\n📁 Checking '{docs_folder}' folder...")
        
        files = os.listdir(docs_folder)
        print(f"  Found {len(files)} files:")
        for f in files:
            print(f"    - {f}")
        
        if files:
            print("\n📖 Attempting to load documents...")
            documents = loader.load_documents_from_folder(docs_folder)
            
            print(f"\n✅ Successfully loaded {len(documents)} documents:")
            for doc in documents:
                print(f"  - {doc['file_name']}: {doc['content_length']:,} characters")
        else:
            print("\n⚠️  No files found in docs folder!")
            print("   Please add some PDF, DOCX, or TXT files")
    else:
        print(f"\n⚠️  Docs folder not found at: {docs_folder}")
        print("   Let's create it...")
        os.makedirs(docs_folder)
        print(f"✓ Created '{docs_folder}' folder")
        print("   Please add some sample files to this folder")
        print("   Then run this test again")

if __name__ == "__main__":
    main()