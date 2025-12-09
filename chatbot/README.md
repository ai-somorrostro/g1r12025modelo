# Ingestión de normas (chatbot)

Este directorio contiene utilidades para extraer y chunkear el PDF de normas (`magic_rules.pdf`) en `jsonl` listo para generar embeddings y alimentar un retriever.

Requisitos:

- Python 3.10+
- Instalar dependencias:

```powershell
pip install -r chatbot/requirements.txt
```

Uso:

```powershell
python chatbot/chunk_ruler.py --input chatbot/magic-rules/magic_rules.pdf --output chatbot/chunks_normas.jsonl
```

Salida:

- `chatbot/chunks_normas.jsonl`: un JSONL con un chunk por encabezado detectado. Cada línea tiene:
  - `id`: identificador del chunk
  - `title`: título o encabezado detectado
  - `text`: contenido del chunk
  - `start_page`, `end_page`: páginas de origen
  - `source`: ruta al PDF

Notas y siguientes pasos:

- El chunking es heurístico (numeración, 'Artículo', líneas en mayúsculas). Se puede ajustar para mejorar la granularidad.
- En el futuro se añadirá un paso que calcule embeddings (OpenRouter/OpenAI/sentence-transformers) y persista en una vector DB (FAISS/Chroma/Pinecone).
- Si quieres, ejecuto el script ahora y te muestro los primeros 10 chunks.

Embeddings

- Instalación de dependencias (recomendado):

```powershell
python -m pip install -r chatbot/requirements.txt
```

Nota: `sentence-transformers` requiere `torch`. En Windows puede ser necesario instalar primero la rueda de CPU de PyTorch (ver https://pytorch.org/get-started/locally/).

- Dry-run (valida que `chatbot/chunks_normas.jsonl` existe sin descargar modelos):

```powershell
python chatbot/embed_chunks.py --input chatbot/chunks_normas.jsonl --dry-run
```

- Generar embeddings y guardar metadatos (crea `chatbot/embeddings.npy` y `chatbot/metadata.jsonl`):

```powershell
python chatbot/embed_chunks.py --input chatbot/chunks_normas.jsonl --out-dir chatbot --batch-size 64
```

- Opcional: crear un índice FAISS (requiere `faiss-cpu`):

```powershell
python -m pip install faiss-cpu
python chatbot/embed_chunks.py --input chatbot/chunks_normas.jsonl --out-dir chatbot
```

 Recomendación de almacenamiento:

 - Guardar `embeddings.npy` (NumPy float32) y `metadata.jsonl` (uno JSON por línea) es la forma recomendada:

 ```powershell
 python chatbot/embed_chunks.py --input chatbot/chunks_normas.jsonl --out-dir chatbot --batch-size 64
 ```

 - Esto producirá `chatbot/embeddings.npy` y `chatbot/metadata.jsonl`.

 - Razones: `npy` es eficiente y permite `mmap` para leer sin cargar todo en memoria; `jsonl` facilita inspeccionar/filtrar metadatos.
