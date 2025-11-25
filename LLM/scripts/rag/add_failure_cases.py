"""
Add failure cases to FAISS index

Phase 3.3.2: Embed synthetic failure cases and expand knowledge base
Expands from 27 machine docs to 127+ total documents
"""
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import json
from pathlib import Path


def add_failure_cases_to_index():
    """Embed and add failure cases"""
    
    print("=" * 60)
    print("PHASE 3.3.2: Embed Failure Cases in FAISS Index")
    print("=" * 60)
    print()
    
    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Model loaded\n")
    
    # Load existing index
    embeddings_dir = Path("../../data/embeddings")
    print(f"Loading existing FAISS index from {embeddings_dir}...")
    index = faiss.read_index(str(embeddings_dir / "machines.index"))
    
    with open(embeddings_dir / "metadata.pkl", 'rb') as f:
        data = pickle.load(f)
        docs = data['docs']
        metadata = data['metadata']
    
    print(f"✓ Loaded existing index: {len(docs)} documents\n")
    
    # Load failure cases
    failure_cases_file = Path("../../data/knowledge_base/failure_cases.json")
    print(f"Loading failure cases from {failure_cases_file}...")
    with open(failure_cases_file) as f:
        cases = json.load(f)
    
    print(f"✓ Loaded {len(cases)} failure cases\n")
    
    # Convert cases to documents
    print("Converting failure cases to text documents...")
    new_docs = []
    new_metadata = []
    
    for i, case in enumerate(cases, 1):
        doc = f"""
FAILURE CASE:
Machine: {case['machine_id']}
Type: {case['failure_type']}
RUL: {case['rul_hours']:.1f} hours
Severity: {case['severity']}

SYMPTOMS:
{chr(10).join(f"- {s}" for s in case['symptoms'])}

ROOT CAUSE: {case['root_cause']}
CORRECTIVE ACTION: {case['corrective_action']}
COST IMPACT: {case['cost_impact']}
"""
        new_docs.append(doc)
        new_metadata.append({
            'type': 'failure_case',
            'machine_id': case['machine_id'],
            'failure_type': case['failure_type']
        })
        
        if i % 25 == 0:
            print(f"  Processed {i}/{len(cases)} cases...")
    
    print(f"✓ Converted {len(new_docs)} cases to documents\n")
    
    # Embed new docs
    print(f"Embedding {len(new_docs)} failure cases (this may take 1-2 minutes)...")
    new_embeddings = model.encode(new_docs, show_progress_bar=True)
    print(f"✓ Embeddings created: {new_embeddings.shape}\n")
    
    # Add to index
    print("Adding embeddings to FAISS index...")
    index.add(new_embeddings.astype('float32'))
    print(f"✓ Added {len(new_embeddings)} embeddings to index\n")
    
    # Update metadata
    print("Updating metadata...")
    docs.extend(new_docs)
    metadata.extend(new_metadata)
    print(f"✓ Total documents now: {len(docs)}\n")
    
    # Save updated index
    print("Saving updated FAISS index...")
    faiss.write_index(index, str(embeddings_dir / "machines.index"))
    with open(embeddings_dir / "metadata.pkl", 'wb') as f:
        pickle.dump({'docs': docs, 'metadata': metadata}, f)
    
    print(f"✓ Index saved to {embeddings_dir}\n")
    
    # Summary
    print("=" * 60)
    print("✅ Phase 3.3.2 Complete!")
    print("=" * 60)
    print(f"\nKnowledge Base Summary:")
    print(f"  - Machine documentation: 27 docs")
    print(f"  - Failure cases: {len(cases)} docs")
    print(f"  - Total: {len(docs)} documents")
    print(f"  - Embedding dimensions: 384")
    print(f"  - Index file: {embeddings_dir / 'machines.index'}")
    print(f"\n📝 Knowledge base ready for RAG retrieval!")


if __name__ == "__main__":
    add_failure_cases_to_index()
