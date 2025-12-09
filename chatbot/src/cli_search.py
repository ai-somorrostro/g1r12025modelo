"""Interactive CLI to search embeddings saved in `embeddings.npy` + `metadata.jsonl`.

Usage:
  Interactive mode (prompt loop):
    .venv\Scripts\python.exe chatbot/cli_search.py --emb chatbot/embeddings.npy --meta chatbot/metadata.jsonl

  One-shot query:
    .venv\Scripts\python.exe chatbot/cli_search.py --emb chatbot/embeddings.npy --meta chatbot/metadata.jsonl --query "¿Qué es girar?" --topk 5

Options:
  --faiss-index to use a prebuilt FAISS index instead of in-memory search.
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


def do_search(emb_path: Path, meta_path: Path, query: str, topk: int = 5, faiss_index: str = None):
    embeddings = np.load(str(emb_path), mmap_mode='r')
    metadata = load_metadata(str(meta_path))

    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        print('sentence-transformers not installed in the environment. Install requirements and retry.')
        raise

    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    q_emb = model.encode([query], show_progress_bar=False)[0].astype('float32')
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)

    if faiss_index:
        try:
            import faiss
            fi = Path(faiss_index)
            if not fi.exists():
                raise FileNotFoundError(fi)
            index = faiss.read_index(str(fi))
            D, I = index.search(q_emb.reshape(1, -1), topk)
            for rank, (i, s) in enumerate(zip(I[0], D[0]), start=1):
                item = metadata[int(i)]
                print(f'{rank}. id={item.get("id")} score={float(s):.4f} title={item.get("title")!r} pages={item.get("start_page")}-{item.get("end_page")}')
            return
        except Exception as e:
            print('FAISS search failed, falling back to in-memory search:', e)

    emb_norm = normalize_rows(np.asarray(embeddings))
    idxs, scores = search_numpy(emb_norm, q_emb, topk=topk)
    for rank, (i, s) in enumerate(zip(idxs, scores), start=1):
        item = metadata[int(i)]
        snippet = (item.get('text') or '').replace('\n', ' ')[:400]
        print(f'{rank}. id={item.get("id")} score={float(s):.4f} title={item.get("title")!r} pages={item.get("start_page")}-{item.get("end_page")}')
        print(f'    {snippet}\n')


def main():
    parser = argparse.ArgumentParser(description='Interactive search for RAG pipeline')
    parser.add_argument('--emb', required=True, help='Path to embeddings.npy')
    parser.add_argument('--meta', required=True, help='Path to metadata.jsonl')
    parser.add_argument('--query', help='One-shot query (if not provided, enter interactive prompt)')
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--faiss-index', help='Optional FAISS index file (auto-detected if not provided)')
    args = parser.parse_args()

    emb_path = Path(args.emb)
    meta_path = Path(args.meta)
    if not emb_path.exists() or not meta_path.exists():
        print('Embeddings or metadata file not found. Check the paths.')
        sys.exit(1)

    # Auto-detect FAISS index if not provided
    faiss_index_path = args.faiss_index
    if not faiss_index_path:
        default_faiss = emb_path.parent / 'faiss.index'
        if default_faiss.exists():
            faiss_index_path = str(default_faiss)
            print(f'Auto-detected FAISS index: {default_faiss}')

    if args.query:
        do_search(emb_path, meta_path, args.query, topk=args.topk, faiss_index=faiss_index_path)
        return

    print('Interactive search. Enter an empty line to exit.')
    while True:
        try:
            q = input('\nPrompt: ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nExiting.')
            break
        if not q:
            print('Exiting.')
            break
        do_search(emb_path, meta_path, q, topk=args.topk, faiss_index=faiss_index_path)


if __name__ == '__main__':
    main()
