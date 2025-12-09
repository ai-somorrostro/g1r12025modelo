"""Simple retrieval from a pickle containing {'embeddings': ndarray, 'metadata': list}.

Usage examples:
  Dry-run (validate pkl):
    .venv\Scripts\python.exe chatbot/retrieve_from_pkl.py --pkl chatbot/embeddings_and_meta.pkl --dry-run

  Run a query (will download model if not present):
    .venv\Scripts\python.exe chatbot/retrieve_from_pkl.py --pkl chatbot/embeddings_and_meta.pkl --query "¿Qué es girar?" --topk 5

Options:
  --faiss-index <path>    Use an existing FAISS index (optional). If not provided, the script computes similarities in-memory.

This script expects embeddings to be a NumPy array of shape (n, d) and metadata to be a list with length n.
"""
import argparse
import pickle
from pathlib import Path
import numpy as np
import sys


def load_pkl(pkl_path: str):
    p = Path(pkl_path)
    if not p.exists():
        raise FileNotFoundError(p)
    with open(p, 'rb') as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or 'embeddings' not in data or 'metadata' not in data:
        raise ValueError('Pickle must be a dict with keys "embeddings" and "metadata"')
    emb = data['embeddings']
    meta = data['metadata']
    emb = np.asarray(emb)
    if emb.ndim != 2:
        raise ValueError('embeddings must be a 2D array')
    if len(meta) != emb.shape[0]:
        raise ValueError('metadata length must match number of embeddings')
    return emb, meta


def normalize(v: np.ndarray):
    # Normalize rows to unit length
    v = v.astype('float32')
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def search_numpy(embeddings: np.ndarray, query_vec: np.ndarray, topk: int = 5):
    # embeddings: (n, d) normalized; query_vec: (d,) normalized
    scores = embeddings.dot(query_vec)
    idx = np.argsort(-scores)[:topk]
    return idx, scores[idx]


def main():
    parser = argparse.ArgumentParser(description='Retrieve top-k chunks from a pickle of embeddings')
    parser.add_argument('--pkl', required=True, help='Path to pickle created by embed_chunks.py')
    parser.add_argument('--query', help='User query text to retrieve')
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--dry-run', action='store_true', help='Only validate the pickle and print counts')
    parser.add_argument('--faiss-index', help='Optional FAISS index file to use for fast search')
    args = parser.parse_args()

    emb, meta = load_pkl(args.pkl)
    print(f'Loaded embeddings: shape={emb.shape}, metadata items={len(meta)}')

    if args.dry_run:
        print('Dry-run: sample metadata (first 5 items):')
        for i, m in enumerate(meta[:5]):
            print(f'  [{i}] id={m.get("id")} title={m.get("title")!r} pages={m.get("start_page")}-{m.get("end_page")}')
        return

    if args.query is None:
        print('Provide --query to perform a retrieval (or use --dry-run to inspect the pickle)')
        sys.exit(1)

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        print('sentence-transformers not available in environment. Install requirements and retry.')
        raise

    model_name = 'sentence-transformers/all-mpnet-base-v2'
    print(f'Loading embedding model {model_name}...')
    model = SentenceTransformer(model_name)
    q_emb = model.encode([args.query], show_progress_bar=False)
    q_emb = np.asarray(q_emb).reshape(-1)

    # Normalize
    emb_norm = normalize(emb)
    q_norm = q_emb.astype('float32')
    q_norm = q_norm / (np.linalg.norm(q_norm) + 1e-12)

    # If user provided a FAISS index path, try to use it
    if args.faiss_index:
        try:
            import faiss
            fi = Path(args.faiss_index)
            if not fi.exists():
                raise FileNotFoundError(fi)
            index = faiss.read_index(str(fi))
            # faiss expects queries of shape (nq, d) float32
            q = q_norm.reshape(1, -1).astype('float32')
            D, I = index.search(q, args.topk)
            # D are inner products if index was built with normalized vectors + IndexFlatIP
            indices = I[0].tolist()
            scores = D[0].tolist()
            print('Top results (FAISS):')
            for rank, (i, s) in enumerate(zip(indices, scores), start=1):
                item = meta[i]
                print(f'{rank}. id={item.get("id")} score={s:.4f} title={item.get("title")!r} pages={item.get("start_page")}-{item.get("end_page")}')
            return
        except Exception as e:
            print('FAISS search failed, falling back to in-memory search:', str(e))

    # In-memory search
    emb_norm = normalize(emb)
    idxs, scores = search_numpy(emb_norm, q_norm, topk=args.topk)
    print('Top results:')
    for rank, (i, s) in enumerate(zip(idxs, scores), start=1):
        item = meta[int(i)]
        snippet = (item.get('text') or '').replace('\n', ' ')[:300]
        print(f'{rank}. id={item.get("id")} score={float(s):.4f} title={item.get("title")!r} pages={item.get("start_page")}-{item.get("end_page")}')
        print(f'    snippet: {snippet!s}\n')


if __name__ == '__main__':
    main()
