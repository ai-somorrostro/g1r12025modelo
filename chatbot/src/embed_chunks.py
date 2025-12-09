import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

def load_chunks(jsonl_path: str) -> List[Dict]:
    chunks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks

def write_metadata(metadata: List[Dict], out_path: str):
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in metadata:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser(description='Generate embeddings for chunks')
    parser.add_argument('--input', '-i', required=True, help='Input JSONL with chunks')
    parser.add_argument('--out-dir', '-o', default='chatbot', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--dry-run', action='store_true', help='Only validate inputs and show counts')
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f'Input file not found: {in_path}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(str(in_path))
    print(f'Chunks loaded: {len(chunks)} from {in_path}')

    if args.dry_run:
        print('Dry-run mode: no model will be loaded. Exiting after validation.')
        return

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        from tqdm import tqdm
    except Exception as e:
        print('Failed to import embedding libraries. Please install requirements:')
        print('  pip install -r chatbot/requirements.txt')
        raise

    model_name = 'sentence-transformers/all-mpnet-base-v2'
    print(f'Loading embedding model {model_name} (this may download model weights)...')
    model = SentenceTransformer(model_name)

    texts = []
    metadata = []
    for c in chunks:
        # Prefer title + text for embeddings
        title = c.get('title') or ''
        text = c.get('text') or ''
        doc_text = (title + '\n' + text).strip()
        texts.append(doc_text)
        metadata.append({
            'id': c.get('id'),
            'title': title,
            'start_page': c.get('start_page'),
            'end_page': c.get('end_page'),
            'source': c.get('source')
        })

    batch_size = args.batch_size
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, show_progress_bar=False)
        embeddings.append(emb)

    import numpy as np
    embeddings = np.vstack(embeddings)

    # Save embeddings as a numpy .npy (float32) and metadata as a JSONL
    import numpy as np
    embeddings = embeddings.astype('float32')
    emb_path = out_dir / 'embeddings.npy'
    meta_path = out_dir / 'metadata.jsonl'
    # Save numpy array
    np.save(str(emb_path), embeddings)
    # Save metadata
    write_metadata(metadata, str(meta_path))
    print(f'Embeddings saved to: {emb_path} (shape={embeddings.shape}, dtype={embeddings.dtype})')
    print(f'Metadata saved to: {meta_path}')

if __name__ == '__main__':
    main()
