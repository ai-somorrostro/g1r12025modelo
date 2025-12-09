r"""RAG system: retrieve context from FAISS, build prompt, call OpenRouter LLM, return answer.

Usage (one-shot):
  .venv\Scripts\python.exe chatbot/rag.py --query "¿Qué es girar?" --topk 5

Usage (interactive):
  .venv\Scripts\python.exe chatbot/rag.py

Environment:
  - Set OPENROUTER_API_KEY in .env or export as env var
  - Models: gpt-4o-mini (fast, cheap), gpt-4-turbo, claude-3-opus, etc.
"""
import argparse
import os
import json
from pathlib import Path
import numpy as np
import sys
from typing import List, Dict


def load_metadata(meta_path: str) -> List[Dict]:
    """Load metadata.jsonl into a list of dicts."""
    items = []
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a single vector (L2)."""
    v = v.astype('float32')
    norm = np.linalg.norm(v)
    return v / (norm + 1e-12)


def retrieve_context(query: str, emb_path: str, meta_path: str, faiss_index_path: str, topk: int = 5) -> List[Dict]:
    """Retrieve top-k chunks similar to query using FAISS or in-memory search."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError('sentence-transformers not installed')

    # Embed query
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    q_emb = model.encode([query], show_progress_bar=False)[0].astype('float32')
    q_emb = normalize_vector(q_emb)

    # Load metadata
    metadata = load_metadata(meta_path)

    # Try FAISS search first
    faiss_path = Path(faiss_index_path)
    if faiss_path.exists():
        try:
            import faiss
            index = faiss.read_index(str(faiss_path))
            D, I = index.search(q_emb.reshape(1, -1), topk)
            results = []
            for i, score in zip(I[0], D[0]):
                item = metadata[int(i)]
                item['score'] = float(score)
                results.append(item)
            return results
        except Exception as e:
            print(f'FAISS search failed: {e}, falling back to in-memory search', file=sys.stderr)

    # Fallback: in-memory search
    embeddings = np.load(emb_path, mmap_mode='r').astype('float32')
    emb_norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    scores = emb_norms.dot(q_emb)
    top_idx = np.argsort(-scores)[:topk]
    results = []
    for i in top_idx:
        item = metadata[int(i)]
        item['score'] = float(scores[i])
        results.append(item)
    return results


def call_openrouter(system_prompt: str, context: str, user_query: str, model: str = 'gpt-4o-mini') -> str:
    """Call OpenRouter LLM with system prompt + context + user query."""
    # The .env should already be loaded by the caller (api.py or main())
    api_key = os.getenv('OPENROUTER_API_KEY')
    
    if not api_key:
        raise ValueError('OPENROUTER_API_KEY not set in environment. Make sure .env is loaded.')

    try:
        import requests
    except ImportError:
        raise ImportError('requests not installed. Install with: pip install requests')

    url = 'https://openrouter.ai/api/v1/chat/completions'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    # Build messages
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': f'Contexto:\n{context}\n\n---\n\nPregunta: {user_query}'}
    ]

    payload = {
        'model': model,
        'messages': messages,
        'temperature': 0.7,
        'max_tokens': 1024,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        if 'choices' in data and len(data['choices']) > 0:
            return data['choices'][0]['message']['content']
        else:
            return f'Error: {data}'
    except Exception as e:
        return f'Error calling OpenRouter: {str(e)}'


def rag_pipeline(query: str, emb_path: str, meta_path: str, faiss_index_path: str, topk: int = 5, model: str = 'gpt-4o-mini') -> str:
    """Full RAG pipeline: retrieve context → call LLM → return answer."""
    
    # System prompt para el RAG
    system_prompt = """Eres un experto en las reglas de Magic: The Gathering. Tu tarea es responder preguntas sobre las reglas basándote en el contexto proporcionado.

Sé preciso, conciso y siempre cita la norma o sección relevante si es posible.
Si el contexto no proporciona la información necesaria, indica que no está disponible en los fragmentos proporcionados."""

    print(f'🔍 Buscando contexto para: "{query}"', file=sys.stderr)
    context_chunks = retrieve_context(query, emb_path, meta_path, faiss_index_path, topk=topk)
    
    # Build context string
    context_str = '\n\n'.join([
        f"[{chunk.get('title', 'Sin título')}] (pages {chunk.get('start_page')}-{chunk.get('end_page')})\n{chunk.get('text', '')}"
        for chunk in context_chunks
    ])
    
    print(f'✓ Encontrado contexto ({len(context_chunks)} chunks)', file=sys.stderr)
    print(f'📞 Llamando a OpenRouter ({model})...', file=sys.stderr)
    
    answer = call_openrouter(system_prompt, context_str, query, model=model)
    return answer


def main():
    # Load .env when running as main script (direct terminal execution)
    from dotenv import load_dotenv
    
    # Try config/.env first, then root .env
    env_path = Path(__file__).parent.parent / 'config' / '.env'
    if not env_path.exists():
        env_path = Path(__file__).parent.parent / '.env'
    
    if env_path.exists():
        load_dotenv(env_path, override=True)
        print(f"[RAG] Loaded .env from: {env_path}", file=sys.stderr)
    else:
        print(f"[RAG] WARNING: .env file not found", file=sys.stderr)
    
    parser = argparse.ArgumentParser(description='RAG chatbot for Magic: The Gathering rules')
    parser.add_argument('--emb', default='chatbot/embeddings.npy', help='Path to embeddings.npy')
    parser.add_argument('--meta', default='chatbot/metadata.jsonl', help='Path to metadata.jsonl')
    parser.add_argument('--faiss-index', default='chatbot/faiss.index', help='Path to FAISS index')
    parser.add_argument('--query', help='One-shot query')
    parser.add_argument('--topk', type=int, default=5, help='Number of context chunks')
    parser.add_argument('--model', default='gpt-4o-mini', help='OpenRouter model (default: gpt-4o-mini)')
    args = parser.parse_args()

    # Validate files
    for fpath in [args.emb, args.meta]:
        if not Path(fpath).exists():
            print(f'File not found: {fpath}')
            sys.exit(1)

    if args.query:
        # One-shot
        answer = rag_pipeline(args.query, args.emb, args.meta, args.faiss_index, topk=args.topk, model=args.model)
        print('\n' + '='*60)
        print(answer)
        print('='*60)
        return

    # Interactive mode
    print('RAG Chatbot - Magic: The Gathering Rules')
    print('Enter an empty line to exit.\n')
    while True:
        try:
            q = input('👤 Pregunta: ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\n👋 Hasta luego!')
            break
        if not q:
            print('👋 Hasta luego!')
            break

        answer = rag_pipeline(q, args.emb, args.meta, args.faiss_index, topk=args.topk, model=args.model)
        print('\n🤖 Respuesta:')
        print(answer)
        print()


if __name__ == '__main__':
    main()
