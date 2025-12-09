"""Retrieve top-k chunks using `embeddings.npy` + `metadata.jsonl`.

Usage:
  Dry-run (validate files):
    .venv\Scripts\python.exe chatbot/retrieve.py --emb chatbot/embeddings.npy --meta chatbot/metadata.jsonl --dry-run

  Query example:
    .venv\Scripts\python.exe chatbot/retrieve.py --emb chatbot/embeddings.npy --meta chatbot/metadata.jsonl --query "¿Qué es girar?" --topk 5

Optional `--faiss-index` to use a prebuilt FAISS index.
"""
import argparse
from pathlib import Path
import numpy as np
import json
import sys


def load_metadata(meta_path: str):
    items = []
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def normalize_rows(v: np.ndarray):
    v = v.astype('float32')
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def search_numpy(embeddings: np.ndarray, query_vec: np.ndarray, topk: int = 5):
    scores = embeddings.dot(query_vec)
    idx = np.argsort(-scores)[:topk]
    return idx, scores[idx]


def main():
    parser = argparse.ArgumentParser(description='Retrieve top-k chunks from embeddings.npy + metadata.jsonl')
    parser.add_argument('--emb', required=True, help='Path to embeddings.npy')
    parser.add_argument('--meta', required=True, help='Path to metadata.jsonl')
    parser.add_argument('--query', help='Text query')
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--faiss-index', help='Optional FAISS index to use for search')
    args = parser.parse_args()

    emb_path = Path(args.emb)
    meta_path = Path(args.meta)
    if not emb_path.exists() or not meta_path.exists():
        print('Embeddings or metadata file not found. Check the paths.')
        sys.exit(1)

    # Load embeddings (use mmap for large arrays)
    embeddings = np.load(str(emb_path), mmap_mode='r')
    metadata = load_metadata(str(meta_path))
    print(f'Loaded embeddings shape={embeddings.shape}, metadata items={len(metadata)}')

    if args.dry_run:
        print('Dry-run: sample metadata (first 5):')
        for i, m in enumerate(metadata[:5]):
            print(f'[{i}] id={m.get("id")} title={m.get("title")!r} pages={m.get("start_page")}-{m.get("end_page")}')
        return

    if args.query is None:
        print('Provide --query to perform retrieval or use --dry-run to inspect files.')
        sys.exit(1)

    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        print('sentence-transformers not installed in the environment. Install requirements and retry.')
        raise

    model_name = 'sentence-transformers/all-mpnet-base-v2'
    print(f'Loading embedding model {model_name}...')
    model = SentenceTransformer(model_name)
    q_emb = model.encode([args.query], show_progress_bar=False)[0].astype('float32')
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)

    # If faiss index provided, try to use it
    if args.faiss_index:
        try:
            import faiss
            fi = Path(args.faiss_index)
            if not fi.exists():
                raise FileNotFoundError(fi)
            index = faiss.read_index(str(fi))
            D, I = index.search(q_emb.reshape(1, -1), args.topk)
            for rank, (i, s) in enumerate(zip(I[0], D[0]), start=1):
                item = metadata[int(i)]
                print(f'{rank}. id={item.get("id")} score={float(s):.4f} title={item.get("title")!r} pages={item.get("start_page")}-{item.get("end_page")}')
            return
        except Exception as e:
            print('FAISS search failed, falling back to in-memory search:', e)

    # In-memory: normalize embeddings rows and compute dot
    emb_norm = normalize_rows(np.asarray(embeddings))
    idxs, scores = search_numpy(emb_norm, q_emb, topk=args.topk)
    print('Top results:')
    for rank, (i, s) in enumerate(zip(idxs, scores), start=1):
        item = metadata[int(i)]
        snippet = (item.get('text') or '').replace('\n', ' ')[:300]
        print(f'{rank}. id={item.get("id")} score={float(s):.4f} title={item.get("title")!r} pages={item.get("start_page")}-{item.get("end_page")}')
        print(f'    snippet: {snippet}\n')


if __name__ == '__main__':
    main()
