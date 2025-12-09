# Docker Setup for MTG RAG Chatbot API

## Con Docker Compose (Recomendado)

### Iniciar el servicio

```bash
cd chatbot
docker compose up -d --build
```

La API estará disponible en `http://localhost:8001`

### Detener el servicio

```bash
docker compose down
```

### Ver logs

```bash
docker compose logs -f mtg-rag-api
```

---

## Con Docker (Alternativa)

### Construir la imagen

```bash
cd chatbot
docker build -t mtg-rag-api:latest .
```

### Ejecutar el contenedor

```bash
docker run -p 8001:8001 mtg-rag-api:latest
```

### Detener el contenedor

```bash
docker stop <container_id>
```

### Ver logs

```bash
docker logs <container_id>
```

---

## Verificar que está funcionando

```bash
curl http://localhost:8001/health
```
