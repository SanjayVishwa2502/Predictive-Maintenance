"""
Generate embeddings for knowledge base using sentence-transformers
Phase 3.1.2: Generate Embeddings
"""
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pathlib import Path
import pickle

def create_faiss_index():
    """Embed all documents and build FAISS index"""
    
    print("Phase 3.1.2: Generating Embeddings for Knowledge Base")
    print("="*60)
    
    # Load embedding model (384-dim, lightweight)
    print("\nLoading sentence-transformers model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Model loaded")
    
    # Load all machine documents
    project_root = Path(__file__).resolve().parents[3]
    docs_dir = project_root / "LLM" / "data" / "knowledge_base" / "machines"
    
    print(f"\nLoading documents from: {docs_dir}")
    docs = []
    metadata = []
    
    for doc_file in sorted(docs_dir.glob("*.txt")):
        text = doc_file.read_text(encoding='utf-8')
        docs.append(text)
        metadata.append({
            'machine_id': doc_file.stem,
            'file': str(doc_file)
        })
    
    print(f"✓ Loaded {len(docs)} documents")
    
    # Create embeddings
    print(f"\nEmbedding {len(docs)} documents...")
    embeddings = model.encode(docs, show_progress_bar=True)
    print(f"✓ Embeddings shape: {embeddings.shape}")
    
    # Create FAISS index
    dimension = embeddings.shape[1]  # 384
    print(f"\nCreating FAISS index (dimension: {dimension})...")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    print(f"✓ FAISS index created with {index.ntotal} vectors")
    
    # Save index and metadata
    output_dir = project_root / "LLM" / "data" / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    index_file = output_dir / "machines.index"
    metadata_file = output_dir / "metadata.pkl"
    
    print(f"\nSaving FAISS index to: {index_file}")
    faiss.write_index(index, str(index_file))
    
    print(f"Saving metadata to: {metadata_file}")
    with open(metadata_file, 'wb') as f:
        pickle.dump({'docs': docs, 'metadata': metadata}, f)
    
    print(f"\n{'='*60}")
    print(f"Phase 3.1.2 Complete!")
    print(f"✓ FAISS index created: {len(docs)} docs, {dimension} dims")
    print(f"✓ Index file: {index_file}")
    print(f"✓ Metadata file: {metadata_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    create_faiss_index()
