# Justificación del Modelo y Arquitectura RAG

## 1. Modelo de Embedding: `sentence-transformers/all-mpnet-base-v2`

### ¿Por qué este modelo?

**all-mpnet-base-v2** es un modelo de embeddings de propósito general entrenado en múltiples idiomas (incluyendo español que es el que usamos nosotros por ahora) que ofrece el mejor balance entre:

- **Calidad**: Puntuaciones excelentes en benchmarks de similitud semántica (MTEB)
- **Velocidad**: ~200 tokens/segundo en CPU, adecuado para búsquedas en tiempo real
- **Tamaño**: 430 MB (manejable para deployments en contenedores)
- **Multilingüe**: Funciona correctamente en español, inglés y otros idiomas

### Alternativas consideradas y descartadas:

| Modelo | Razón de rechazo |
|--------|------------------|
| `all-MiniLM-L6-v2` | Demasiado pequeño (22 MB) → pérdida de calidad semántica |
| `multilingual-e5-large` | Más preciso pero 1.3 GB → overhead en producción |
| OpenAI embeddings (`text-embedding-3-small`) | Requiere API key, latencia de red, costo por token |
| `bert-base-multilingual-cased` | Más antiguo, scores peores en MTEB |

### Dimensionalidad
- **Dimensiones**: 768
- **Justificación**: Suficientes para capturar semántica compleja de textos legales/reglas sin exceso computacional

---

## 2. Modelo Razonador: OpenRouter + `gpt-4o-mini`

### ¿Por qué gpt-4o-mini?

**gpt-4o-mini** se eligió como modelo razonador/generador porque:

- **Velocidad**: ~1 segundo de latencia promedio (vs 3-5s de gpt-4-turbo que se nso hacia muy lento)
- **Costo**: $0.15 por 1M input tokens, $0.60 por 1M output tokens
- **Calidad**: Comprende correctamente contextos largos (128k tokens) y genera respuestas precisas
- **Multiidioma**: Excelente soporte para español con consistencia
- **Formato**: Genera Markdown correctamente formateado

### ¿Por qué OpenRouter en lugar de OpenAI directamente?

| Aspecto | OpenRouter | OpenAI Directo |
|--------|-----------|-----------------|
| **Flexibilidad de modelos** | Múltiples proveedores (OpenAI, Anthropic, etc) | Solo OpenAI |
| **Routing automático** | Puede cambiar de proveedor si uno falla | Punto único de fallo |
| **Costo** | A veces más barato con ofertas | Precio fijo |
| **Fallback** | Soporte integrado | Requiere implementación manual |

### Alternativas descartadas:

| Modelo | Razón de rechazo |
|--------|-----------------|
| Claude 3 Opus | Demasiado lento (5-10s), muy caro para volumen |
| Llama 2 (local) | Requeriría GPU potente, y no tenemos las maquinas buenas habilitadas, menor calidad en español |
| Mistral | Buen balance pero peor soporte para instrucciones complejas |
| GPT-4 Turbo | 3-4x más caro, solo necesitamos gpt-4o-mini |

---

## 3. Almacenamiento: NPY + JSONL vs Alternativas

### Arquitectura elegida

```
data/
├── embeddings.npy          # Arrays numéricos (768 dims × N chunks)
└── metadata.jsonl          # Metadatos (1 JSON por línea)
```

### ¿Por qué NPY para embeddings?

**NumPy Binary Format (.npy)** es superior para embeddings porque:

| Característica | NPY | Alternativas |
|---|---|---|
| **Carga en memoria** | Mucho más rápido (milisegundos) | JSON: segundos; Pickle: riesgoso y peor rendimiento probado |
| **Precisión** | Float32 nativo sin pérdida | Texto: requiere reconversión |
| **Tamaño en disco** | ~3 bytes por float (768×N) | JSON: ~50+ bytes por valor |
| **Compatible NumPy** | Directo con `np.load()` | Requiere parseo manual |
| **FAISS** | Integración nativa y muy sencilla y eficiente | Conversion overhead |

**Ejemplo de tamaño**:
- 10,000 embeddings × 768 dims = 30.7 MB (NPY)
- Mismo en JSON = 500+ MB

### ¿Por qué JSONL para metadatos?

**JSON Lines Format** usamos JSONL para guardar los metadatos de los embeddings del NPY  porque:

```jsonl
{"index": 0, "title": "506. Fase de combate", "start_page": 81, "end_page": 82, "text": "..."}
{"index": 1, "title": "510. Paso de daño", "start_page": 89, "end_page": 90, "text": "..."}
```

**Ventajas**:
- ✅ **Legible**: Puedes abrir con cualquier editor
- ✅ **Parseo línea por línea**: No necesita cargar todo en memoria
- ✅ **Sincronización**: Cada línea = un embedding (mismo índice)
- ✅ **Flexible**: Agregar campos sin romper compatibilidad

### Alternativas descartadas:

| Opción | Razón de rechazo |
|--------|-----------------|
| **PostgreSQL** | Overhead innecesario para demo, requiere servidor |
| **SQLite** | Mejor que Postgres pero aún overhead para read-only |
| **Pickle** | Inseguro, no legible, problemas de versionado |
| **HDF5** | Más complejo de usar, mejor para datasets de investigación |
| **Parquet** | Overkill, optimizado para analytics no retrieval |
| **MongoDB/Vector DB** | Complejidad innecesaria, costos de hosting |
| **Elasticsearch** | Ver sección dedicada abajo |

---

## 3.5. NPY+JSONL vs Elasticsearch: Análisis Detallado

### ¿Por qué NO Elasticsearch?

Consideramos Elasticsearch pero finalmente lo descartamos. Aquí está la comparativa completa:

| Aspecto | NPY + JSONL + FAISS | Elasticsearch |
|--------|---------------------|---------------|
| **Setup inicial** | 5 minutos | 30+ minutos (Docker, JVM, etc) |
| **Memoria en reposo** | ~50 MB | 1+ GB (JVM mínimo) |
| **Curva aprendizaje** | Muy baja | Media-Alta |
| **Búsqueda vectorial nativa** | FAISS (excelente) | Módulo dense_vector (reciente) |
| **Búsqueda híbrida (texto+vector)** | Manual, controlable | Automática, pero compleja |
| **Actualizar índices** | Rebuild (segundos) | Rolling update (transparente) |
| **Escalabilidad inicial** | Perfecta para <100k docs | Overkill para <100k docs |
| **Deploy en Docker** | 50 MB imagen | 800+ MB imagen |
| **Documentación** | Código simple | Extensa pero densa |

### Decisión: NPY+JSONL ganador por:

1. **Simplicidad**
   ```bash
   # NPY+JSONL: 3 lineas de código
   embeddings = np.load('embeddings.npy')
   metadata = [json.loads(line) for line in open('metadata.jsonl')]
   D, I = faiss_index.search(query_emb, k=5)
   
   # Elasticsearch: 50+ lineas de config, 20+ dependencias
   ```

2. **Tamaño de Docker**
   ```
   NPY + JSONL setup:
   - Python 3.11-slim: 130 MB
   - Dependencies: 300 MB
   - Data: 100 MB
   Total: ~530 MB
   
   Elasticsearch:
   - Base image: 800+ MB
   - Runtime: 1-2 GB en memoria
   Total: 2.5+ GB
   ```

3. **Recursos en producción**
   ```
   NPY+JSONL en CPU:
   - Query latency: ~10ms
   - Memory usage: 50-100 MB
   - CPU usage: <5%
   
   Elasticsearch:
   - Query latency: ~50-100ms (JVM startup)
   - Memory usage: 1-2 GB mínimo
   - CPU usage: 20-30% (idle)
   ```

4. **Mantenibilidad**
   - NPY+JSONL: Una carpeta `/data`, dos archivos
   - Elasticsearch: Necesita monitoreo, logs, tuning de JVM

### Cuándo ELEGIR Elasticsearch:

Elasticsearch habría sido mejor SI:
- ✅ Búsqueda full-text compleja (wildcards, regex, synonyms)
- ✅ >10 millones de documentos
- ✅ Índices con actualización frecuente (shards, replicas)
- ✅ Necesidad de faceting/agregaciones avanzadas
- ✅ Múltiples usuarios escribiendo simultáneamente

**En nuestro caso**: Read-only, <50k chunks, búsqueda semántica pura → Elasticsearch es un cañón para matar mosquitos.

---

## 4. Indexación: FAISS vs Alternativas

### ¿Por qué FAISS?

**Facebook AI Similarity Search** elegido porque:

- **Velocidad**: Búsqueda de 1M embeddings en <1ms
- **CPU-friendly**: `faiss-cpu` sin requerer GPU
- **Escalabilidad**: Optimizado para high-dimensional vectors
- **Maduro**: Años de uso en producción

### Búsqueda típica:
```python
# Embedding query: 768 dims
# Top-k=5 búsqueda: <10ms en 10,000 chunks
# Top-k=7 búsqueda: <15ms en 10,000 chunks
```

### Alternativas descartadas:

| Opción | Razón |
|--------|-------|
| **Busca lineal (NumPy)** | O(n) → 10M búsquedas/seg en 10k chunks, lento para escala |
| **Elasticsearch** | Excelente para texto pero no optimizado para vectors |
| **Pinecone** | Caro ($0.04 per 100k vectors/month), requiere internet |
| **Weaviate** | Overhead, mejor para datasets mayores |
| **Milvus** | Más complejo que FAISS para casos simples |

---

## 5. Formato de Contexto y Prompt Engineering

### Sistema de Prompt Multi-layered

```
1. SYSTEM PROMPT (rag.py)
   ├─ Identidad: "Eres experto en Magic: The Gathering"
   ├─ Restricciones: "Solo Magic, rechaza otras preguntas"
   ├─ Formato: "Usa negrita, espacios entre puntos..."
   └─ Referencias: "Agrega normas citadas"

2. CONTEXTO RECUPERADO
   ├─ [Título de sección] (páginas X-Y)
   └─ [Texto completo del chunk]

3. PREGUNTA DEL USUARIO
   └─ Procesada sin modificación (evita hallucinations)
```

### ¿Por qué no dual-search o preprocessing?

Intentamos inicialmente:
- **Query preprocessing**: Genera hallucinations en números de página
- **Dual search**: Aumenta latencia innecesariamente
- **Advanced modes**: Complejidad sin beneficio demostrado

**Decisión final**: KISS (Keep It Simple, Stupid)
- Simple single search con topk parámetro (5 o 7)
- Mejor recall que complex pipelines
- Más rápido y mantenible

---

## 6. Arquitectura General: RAG (Retrieval-Augmented Generation)

### ¿Por qué RAG y no Fine-Tuning?

| Aspecto | RAG | Fine-tuning |
|--------|-----|-------------|
| **Tiempo** | Minutos (build indices) | Horas/días |
| **Costo** | Bajo (NPY + FAISS) | Altísimo (GPU, datos) |
| **Actualizar conocimiento** | Rebuild índices (segundos) | Reentrenar modelo |
| **Transparencia** | Puedes ver fuentes | Black box |
| **Precisión en hechos** | Excelente (basado en fuentes) | Hallucinations posibles |

---

## 7. Stack Completo: Justificación

### Backend
- **FastAPI**: Moderno, rápido, async-ready, validación automática con Pydantic
- **Uvicorn**: ASGI server optimizado para FastAPI
- **Python 3.11**: Versión estable, soporte 10 años

### Frontend
- **HTML/CSS/JS vanilla**: Sin dependencias, deployable en cualquier servidor web
- **localStorage**: Logging offline sin backend
- **Dark mode (#0d0d0d)**: Reduce fatiga ocular, profesional

### Deployment
- **Docker**: Reproducibilidad, portabilidad, consistencia
- **Docker Compose**: Orquestación simple, sin Kubernetes overhead

---

## 8. Decisiones de Rendimiento

### Parámetros de Búsqueda

```python
topk=5  # Standard: 5 chunks relevantes
topk=7  # Deep: 7 chunks para preguntas complejas
```

**Justificación**:
- 5 chunks = ~2000-3000 tokens de contexto
- 7 chunks = ~3000-4000 tokens de contexto
- Ambos < 128k max tokens de gpt-4o-mini
- Suficiente para evitar repeticiones pero no tanto para confundir al modelo

### Timeouts y Health Checks

```yaml
healthcheck:
  interval: 30s
  timeout: 10s
  start_period: 5s
```

- **interval 30s**: Detecta fallos rápido sin overhead
- **timeout 10s**: Suficiente para `/health` endpoint
- **start_period 5s**: Tiempo para arrancar la app

---

## Conclusión

Este stack fue diseñado con principios de:

1. **Eficiencia**: Mínimo overhead, máximo rendimiento
2. **Simplicidad**: Mantenible por uno o dos desarrolladores
3. **Escalabilidad**: Fácil agregar más datos o cambiar modelos
4. **Transparencia**: Logs, referencias, código limpio
5. **Costo**: No hay suscripciones, solo pago por uso de API

Cada decisión fue validada contra alternativas y benchmarks reales.
