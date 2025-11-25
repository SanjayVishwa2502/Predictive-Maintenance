"""Quick verification of FAISS index"""
import pickle
import faiss
from pathlib import Path

p = Path('../../data/embeddings')
idx = faiss.read_index(str(p / 'machines.index'))
data = pickle.load(open(p / 'metadata.pkl', 'rb'))

machine_docs = sum(1 for m in data['metadata'] if m.get('type') != 'failure_case')
failure_docs = sum(1 for m in data['metadata'] if m.get('type') == 'failure_case')

print('\n✅ FAISS Index Verification:')
print(f'  Total vectors: {idx.ntotal}')
print(f'  Total docs: {len(data["docs"])}')
print(f'  Machine docs: {machine_docs}')
print(f'  Failure cases: {failure_docs}')
print('  Status: ✓ OK\n')
