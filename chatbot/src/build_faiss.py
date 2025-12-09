"""Build a FAISS index from embeddings.npy for fast retrieval.

Usage:
  .venv\Scripts\python.exe chatbot/build_faiss.py --emb chatbot/embeddings.npy --output chatbot/faiss.index

Output:
  - chatbot/faiss.index: serialized FAISS IndexFlatIP
"""
import argparse
import numpy as np
from pathlib import Path


def build_faiss_index(emb_path: str, output_path: str):
    try:
        import faiss
    except ImportError:
        print('faiss not installed. Install with: pip install faiss-cpu')
        raise

    emb = np.load(emb_path).astype('float32')
    print(f'Loaded embeddings shape={emb.shape}')

    # Normalize for cosine similarity (IndexFlatIP = inner product on normalized vectors = cosine)
    faiss.normalize_L2(emb)
    print('Normalized embeddings (L2)')

    # Build IndexFlatIP (exact, fast for < 1M vectors)
    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb)
    print(f'Built IndexFlatIP, ntotal={index.ntotal}')

    # Save
    faiss.write_index(index, output_path)
    print(f'Saved FAISS index to {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Build FAISS index from embeddings')
    parser.add_argument('--emb', required=True, help='Path to embeddings.npy')
    parser.add_argument('--output', '-o', default='chatbot/faiss.index', help='Output FAISS index file')
    args = parser.parse_args()

    if not Path(args.emb).exists():
        print(f'Embeddings file not found: {args.emb}')
        return

    build_faiss_index(args.emb, args.output)


if __name__ == '__main__':
    main()
