"""Test script to check if vector store has documents."""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.vectorstore import VectorStore

try:
    # Initialize vector store
    vectorstore = VectorStore()
    
    # Get document count
    doc_count = vectorstore.get_document_count()
    
    print(f"✅ Vector Store Status:")
    print(f"   Documents in store: {doc_count}")
    
    if doc_count > 0:
        print(f"✅ Documents found! Ready to query.")
    else:
        print(f"⚠️  No documents found. Please upload documents first.")
        print(f"   Either:")
        print(f"   1. Upload via Streamlit UI (Document Management tab)")
        print(f"   2. Run: python scripts/ingest_documents.py <files>")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
