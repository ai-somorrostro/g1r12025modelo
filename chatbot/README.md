# Magic: The Gathering RAG Chatbot

Un sistema completo de **Retrieval-Augmented Generation (RAG)** para consultar las reglas de Magic: The Gathering. Incluye CLI interactivo y API REST con FastAPI.

## 🚀 Características

- **Extracción inteligente de chunks** desde PDF de normas
- **Embeddings semánticos** usando `sentence-transformers`
- **Búsqueda rápida** con FAISS
- **RAG completo**: recupera contexto + LLM (OpenRouter) = respuestas precisas
- **CLI interactivo** para consultas directas
- **API REST** con FastAPI + Swagger UI

## 📁 Estructura del Proyecto

```
chatbot/
├── src/                    # Scripts de la cadena de procesamiento
│   ├── chunk_ruler.py     # Extrae chunks del PDF
│   ├── embed_chunks.py    # Genera embeddings
│   ├── build_faiss.py     # Crea índice FAISS
│   ├── cli_search.py      # Búsqueda CLI pura (sin LLM)
│   └── rag.py             # RAG completo (retrieval + LLM)
├── data/                   # Artefactos generados
│   ├── pdf/               # PDFs fuente
│   ├── chunks_normas.jsonl
│   ├── embeddings.npy
│   ├── metadata.jsonl
│   └── faiss.index
├── config/                 # Configuración
│   └── .env               # Variables de entorno (API keys)
├── docs/                   # Documentación adicional
├── api.py                 # Servidor FastAPI
├── requirements.txt       # Dependencias Python
└── .gitignore            # Exclusiones para git
```

## 🛠️ Configuración Inicial

### 1. Crear el entorno virtual

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Instalar dependencias

```powershell
pip install -r requirements.txt
```

### 3. Configurar variables de entorno

Crea `config/.env` con tu API key de OpenRouter:

```env
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxx
```

## 🔄 Pipeline: Cómo Funciona

### Fase 1: Preparación de datos (una sola vez)

```powershell
# 1. Extraer chunks del PDF
python src/chunk_ruler.py --input "data/pdf/magic_rules.pdf" --output "data/chunks_normas.jsonl"

# 2. Generar embeddings
python src/embed_chunks.py --input "data/chunks_normas.jsonl" --out-dir "data"

# 3. Construir índice FAISS
python src/build_faiss.py --emb "data/embeddings.npy" --output "data/faiss.index"
```

### Fase 2: Usar el RAG

#### Opción A: CLI interactivo

```powershell
python src/rag.py --emb "data/embeddings.npy" --meta "data/metadata.jsonl" --faiss-index "data/faiss.index"
```

Luego escribe tus preguntas:
```
👤 Pregunta: ¿Qué es girar?
🤖 Respuesta: [Respuesta generada por LLM]
```

#### Opción B: CLI de una sola pregunta

```powershell
python src/rag.py --emb "data/embeddings.npy" --meta "data/metadata.jsonl" --faiss-index "data/faiss.index" --query "¿Qué es girar?"
```

#### Opción C: API REST

```powershell
# Terminal 1: Iniciar servidor API
python api.py
# Navega a http://localhost:8001/docs para Swagger UI
```

#### Opción D: Web Interface (Recomendado)

La forma más fácil y amigable de usar el chatbot:

**Terminal 1: Iniciar el API**
```powershell
python api.py
```

**Terminal 2: Servir la interfaz web**
```powershell
python serve_web.py
```

Luego abre: **http://localhost:8000**

O simplemente abre `web/index.html` en tu navegador.

## 🌐 Web Interface

La interfaz web está en la carpeta `web/`. Es una UI moderna estilo ChatGPT con:

- Chat limpio y simple (sin historial)
- Selector de modelos LLM
- Control de parámetros (topk)
- Indicador de carga
- Responsive design

**Archivos:**
- `web/index.html` - Estructura HTML
- `web/style.css` - Estilos (diseño moderno)
- `web/script.js` - Lógica del cliente
- `web/README.md` - Documentación detallada

Ver `web/README.md` para más detalles.

## 📡 API Endpoints

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Info del API |
| `/health` | GET | Health check |
| `/query` | POST | Hacer pregunta RAG |
| `/docs` | GET | Swagger UI |

### POST /query

**Request:**
```json
{
  "query": "¿Qué es girar?",
  "topk": 5,
  "model": "gpt-4o-mini"
}
```

**Response:**
```json
{
  "query": "¿Qué es girar?",
  "answer": "Girar es...",
  "topk": 5,
  "model": "gpt-4o-mini"
}
```

## 🔑 Configuración Avanzada

### Variables de entorno
- `OPENROUTER_API_KEY`: Tu API key de OpenRouter (requerido)

### Modelos disponibles en OpenRouter
- `gpt-4o-mini` (rápido, barato) ⭐ recomendado
- `gpt-4-turbo` (más potente)
- `claude-3-opus` (alternativa)
- Más en https://openrouter.ai/

## 🧪 Testing

Test del API completo:
```powershell
python test_api.py
```

Test de búsqueda simple (sin LLM):
```powershell
python src/cli_search.py --emb "data/embeddings.npy" --meta "data/metadata.jsonl" --faiss-index "data/faiss.index"
```

## 🐛 Troubleshooting

### "OPENROUTER_API_KEY not set"
- Verifica que `config/.env` existe y tiene la API key correcta
- Verifica que el archivo NO está en formato UTF-8 con BOM

### API no responde
- Asegúrate que estás en la carpeta `chatbot/` cuando ejecutas `python api.py`
- Verifica que el puerto 8001 no está siendo usado por otro proceso

### Embeddings lentos
- Es normal en la primera ejecución (descarga modelo de ~400MB)
- Las ejecuciones posteriores son mucho más rápidas (modelo en caché)

## 📝 Notas de Arquitectura

- **`cli_search.py`**: Solo retrieval FAISS, sin LLM. Útil para debug
- **`rag.py`**: Retrieval + LLM. Respuestas de calidad
- **`.env` loading**: El API carga `.env` al iniciar. Los scripts CLI lo cargan en `main()`
- **Paths relativos**: El API debe ejecutarse desde la carpeta `chatbot/`

## 🚀 Próximos Pasos

- Agregar caché de respuestas para preguntas frecuentes
- Mejorar chunking con semántica (vs. heurísticos)
- Soportar múltiples PDFs
- Docker containerization
- Autenticación API
- Logging y monitoring

## 📄 Licencia

Este proyecto usa datos públicos de Magic: The Gathering desde Scryfall.

---

**Preguntas?** Revisa la documentación en `docs/README.md` o los scripts individuales.
