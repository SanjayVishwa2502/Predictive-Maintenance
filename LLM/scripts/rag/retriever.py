"""
RAG retrieval system for machine documentation
Phase 3.1.3: Build Retrieval System
"""
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path
import time

class MachineDocRetriever:
    def __init__(self):
        print("Initializing Machine Document Retriever...")
        
        # Load embedding model
        print("Loading sentence-transformers model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("[OK] Model loaded")
        
        # Load FAISS index
        project_root = Path(__file__).resolve().parents[3]
        embeddings_dir = project_root / "LLM" / "data" / "embeddings"
        
        print(f"Loading FAISS index from: {embeddings_dir}")
        self.index = faiss.read_index(str(embeddings_dir / "machines.index"))
        print(f"[OK] FAISS index loaded ({self.index.ntotal} vectors)")
        
        # Load metadata
        print("Loading document metadata...")
        with open(embeddings_dir / "metadata.pkl", 'rb') as f:
            data = pickle.load(f)
            self.docs = data['docs']
            self.metadata = data['metadata']
        print(f"[OK] Metadata loaded ({len(self.docs)} documents)")
        
        print("[OK] Retriever ready!\n")
    
    def retrieve(self, query, machine_id=None, top_k=3):
        """
        Retrieve relevant docs for query
        
        Args:
            query: Natural language question
            machine_id: Optional filter for specific machine
            top_k: Number of docs to return
        
        Returns:
            List of dicts with 'doc', 'machine_id', and 'score'
        """
        start_time = time.time()
        
        # Embed query
        query_emb = self.model.encode([query])
        
        # Search FAISS (get more candidates if filtering by machine)
        search_k = min(top_k * 10 if machine_id else top_k, self.index.ntotal)
        distances, indices = self.index.search(query_emb.astype('float32'), search_k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= len(self.docs):  # Safety check
                continue
                
            doc = self.docs[idx]
            meta = self.metadata[idx]
            
            # Filter by machine if specified
            if machine_id and meta['machine_id'] != machine_id:
                continue
            
            results.append({
                'doc': doc,
                'machine_id': meta['machine_id'],
                'score': float(dist)
            })
            
            # Stop when we have enough results
            if len(results) >= top_k:
                break
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return results, elapsed_ms

# Test retrieval
if __name__ == "__main__":
    print("="*60)
    print("Phase 3.1.3: Testing Retrieval System")
    print("="*60 + "\n")
    
    # Initialize retriever
    retriever = MachineDocRetriever()
    
    # Test 1: General query without machine filter
    print("TEST 1: General Query (No Machine Filter)")
    print("-"*60)
    query1 = "What are common bearing failures?"
    print(f"Query: '{query1}'")
    print(f"Top-K: 3\n")
    
    results1, latency1 = retriever.retrieve(
        query=query1,
        top_k=3
    )
    
    print(f"Retrieved {len(results1)} documents in {latency1:.2f}ms\n")
    
    for i, r in enumerate(results1):
        print(f"=== Result {i+1} ===")
        print(f"Machine: {r['machine_id']}")
        print(f"Score: {r['score']:.4f}")
        print(f"Content Preview:\n{r['doc'][:250]}...\n")
    
    # Test 2: Query with machine filter
    print("\n" + "="*60)
    print("TEST 2: Query with Machine Filter")
    print("-"*60)
    query2 = "What sensors are monitored?"
    machine2 = "motor_siemens_1la7_001"
    print(f"Query: '{query2}'")
    print(f"Machine ID: {machine2}")
    print(f"Top-K: 2\n")
    
    results2, latency2 = retriever.retrieve(
        query=query2,
        machine_id=machine2,
        top_k=2
    )
    
    print(f"Retrieved {len(results2)} documents in {latency2:.2f}ms\n")
    
    for i, r in enumerate(results2):
        print(f"=== Result {i+1} ===")
        print(f"Machine: {r['machine_id']}")
        print(f"Score: {r['score']:.4f}")
        print(f"Content Preview:\n{r['doc'][:250]}...\n")
    
    # Test 3: Maintenance-related query
    print("\n" + "="*60)
    print("TEST 3: Maintenance Query")
    print("-"*60)
    query3 = "Maintenance procedures and safety notes"
    print(f"Query: '{query3}'")
    print(f"Top-K: 3\n")
    
    results3, latency3 = retriever.retrieve(
        query=query3,
        top_k=3
    )
    
    print(f"Retrieved {len(results3)} documents in {latency3:.2f}ms\n")
    
    for i, r in enumerate(results3):
        print(f"=== Result {i+1} ===")
        print(f"Machine: {r['machine_id']}")
        print(f"Score: {r['score']:.4f}\n")
    
    # Performance Summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    avg_latency = (latency1 + latency2 + latency3) / 3
    print(f"Average Retrieval Latency: {avg_latency:.2f}ms")
    print(f"Target: <100ms")
    print(f"Status: {'✓ PASS' if avg_latency < 100 else '✗ FAIL'}")
    
    print("\n" + "="*60)
    print("Phase 3.1.3 Complete: Retrieval System Working!")
    print("="*60)
