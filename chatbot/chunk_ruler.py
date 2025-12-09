#!/usr/bin/env python3
"""
Script para extraer y chunkear el PDF de normas por encabezados (normas, subnormas, subapartados).

Output: un archivo JSONL con un chunk por norma/subnorma/subapartado listo para embeddear.

Uso:
  python ingest_normas.py --input chatbot/magic-rules/magic_rules.pdf --output chatbot/chunks_normas.jsonl

Notas:
- Heurísticas de detección de encabezados:
  * líneas que empiezan con numeración: 1., 1.1, 1.1.1, etc.
  * líneas que contienen 'Artículo', 'Capítulo', 'Sección', 'Norma' (case-insensitive)
  * líneas en MAYÚSCULAS (longitud mínima)

Este script no realiza embeddings; genera `jsonl` con campos: id, title, text, start_page, end_page, source
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict

import fitz  # PyMuPDF


HEADING_PATTERNS = [
    re.compile(r'^\s*(\d+(?:\.\d+){0,})[\.)]?\s+(.+)$'),  # 1. 1.1 1.1.1
    re.compile(r'^(?:Artículo|Capítulo|Sección)\b\s*\d*', re.IGNORECASE),
    re.compile(r'^(?:Norma|Subnorma)\b', re.IGNORECASE),
]


def is_uppercase_heading(line: str) -> bool:
    s = line.strip()
    return len(s) >= 4 and s == s.upper() and any(c.isalpha() for c in s)


def detect_heading(line: str) -> bool:
    if not line or len(line.strip()) == 0:
        return False
    for p in HEADING_PATTERNS:
        if p.search(line):
            return True
    if is_uppercase_heading(line):
        return True
    return False


def detect_definition_term(line: str) -> bool:
    """
    Heurística para detectar un "término" en la sección de definiciones.

    Reglas utilizadas:
    - Línea corta (<= 6 palabras)
    - No contiene punto final ni dos puntos (no es oración completa)
    - No coincide con encabezado numérico (no es una norma)
    - La primera letra es mayúscula (típico en términos: "Mundo", "Ciclo de Hechicero")
    - Longitud entre 1 y 80 caracteres
    """
    if not line:
        return False
    s = line.strip()
    if len(s) == 0 or len(s) > 80:
        return False
    # Evitar frases largas o que terminen con punto/colon
    if '.' in s or ':' in s:
        return False
    words = s.split()
    if len(words) > 6:
        return False
    # No numeración (evitar normas con 100.1 etc.)
    if HEADING_PATTERNS[0].search(s):
        return False
    # Requerir que empiece por mayúscula
    if not s[0].isalpha() or not s[0].isupper():
        return False
    # Evitar líneas que son exactamente una sola palabra seguida de coma etc
    return True


def extract_chunks_from_pdf(pdf_path: Path, allowed_pages: List[int] = None, definitions_start_page: int = None) -> List[Dict]:
    doc = fitz.open(str(pdf_path))
    chunks = []

    current_chunk_lines: List[str] = []
    current_title = None
    start_page = 0
    last_processed_page = None

    def flush_chunk(end_page):
        nonlocal current_chunk_lines, current_title, start_page
        if not current_chunk_lines and not current_title:
            return
        text = "\n".join([l.rstrip() for l in current_chunk_lines]).strip()
        if not text and not current_title:
            return
        chunk = {
            "id": f"chunk_{len(chunks)+1}",
            "title": current_title or "Sin título",
            "text": text,
            "start_page": start_page + 1,
            "end_page": end_page + 1,
            "source": str(pdf_path)
        }
        chunks.append(chunk)
        current_chunk_lines = []
        current_title = None

    for page_num in range(len(doc)):
        if allowed_pages is not None and page_num not in allowed_pages:
            continue
        page = doc.load_page(page_num)
        # Si la página actual no es consecutiva a la última procesada, cerramos el chunk en curso
        if last_processed_page is not None and page_num != last_processed_page + 1:
            flush_chunk(last_processed_page)
        # registramos la última página procesada de las permitidas
        last_processed_page = page_num
        page_text = page.get_text("text")
        lines = page_text.splitlines()

        for i, line in enumerate(lines):
            # Si estamos en la zona de definiciones (1-indexed compare)
            in_definitions = definitions_start_page is not None and (page_num + 1) >= definitions_start_page

            if in_definitions:
                # Modo definiciones: detectamos término corto + su definición (las siguientes líneas)
                if detect_definition_term(line):
                    # cerramos chunk previo
                    if current_chunk_lines:
                        flush_chunk(last_processed_page if last_processed_page is not None else page_num)
                    current_title = line.strip()
                    start_page = page_num
                    # arrancamos colección de la definición (las siguientes líneas)
                    current_chunk_lines = []
                    continue
                else:
                    # acumular texto de definición hasta que aparezca el siguiente término
                    if current_title is None and not current_chunk_lines:
                        # bloque previo a primer término en la sección de definiciones
                        start_page = page_num
                        current_chunk_lines.append(line)
                    else:
                        current_chunk_lines.append(line)
            else:
                # Modo normas (original)
                if detect_heading(line):
                    # Cuando encontramos un heading, cerramos chunk actual y empezamos uno nuevo
                    # Si no había título antes, usamos esta línea como título
                    if current_chunk_lines:
                        flush_chunk(last_processed_page if last_processed_page is not None else page_num)

                    current_title = line.strip()
                    start_page = page_num
                    # No añadimos el heading a `current_chunk_lines` para que title quede limpio,
                    # pero añadimos la línea siguiente si existe (será parte del contenido).
                    continue
                else:
                    # Si no hay título aún, acumulamos pero sin crear chunk hasta encontrar heading
                    if current_title is None and not current_chunk_lines:
                        # Empieza un bloque previo a primer heading: lo acumulamos y lo pondremos
                        # como chunk si no se detecta ningún heading. Fijamos start_page a la
                        # página actual para que las páginas de origen sean correctas.
                        start_page = page_num
                        current_chunk_lines.append(line)
                    else:
                        current_chunk_lines.append(line)

    # Al final del documento, vaciamos el último chunk usando la última página procesada
    flush_chunk(last_processed_page if last_processed_page is not None else (len(doc) - 1))
    return chunks


def write_jsonl(chunks: List[Dict], out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Extrae y chunkea un PDF de normas.")
    parser.add_argument("--input", "-i", required=True, help="Ruta al PDF de normas")
    parser.add_argument("--output", "-o", required=False, default="chatbot/chunks_normas.jsonl", help="Archivo JSONL de salida")
    parser.add_argument("--pages", "-p", required=False, default=None,
                        help="Rangos de páginas a procesar, e.g. '6-10,325-330' (1-indexed)")
    parser.add_argument("--definitions-start", required=False, default=None, type=int,
                        help="(opcional) página (1-indexed) a partir de la cual el documento pasa a ser 'definiciones' y se usa el modo de chunking de definiciones")
    args = parser.parse_args()

    pdf_path = Path(args.input)
    out_path = Path(args.output)

    if not pdf_path.exists():
        print(f"Error: no existe el archivo {pdf_path}")
        return

    print(f"Procesando {pdf_path} ...")

    # Parse page ranges if provided (user uses 1-indexed pages)
    allowed_pages = None
    if args.pages:
        allowed_pages = set()
        for part in args.pages.split(','):
            part = part.strip()
            if '-' in part:
                a, b = part.split('-', 1)
                try:
                    a_i = int(a)
                    b_i = int(b)
                except ValueError:
                    continue
                # convert to 0-indexed pages and add
                for pg in range(max(1, a_i), b_i + 1):
                    allowed_pages.add(pg - 1)
            else:
                try:
                    pg = int(part)
                    allowed_pages.add(pg - 1)
                except ValueError:
                    continue
        allowed_pages = sorted(allowed_pages)

    definitions_start = int(args.definitions_start) if args.definitions_start else None
    chunks = extract_chunks_from_pdf(pdf_path, allowed_pages=allowed_pages, definitions_start_page=definitions_start)

    if not chunks:
        print("No se generaron chunks.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(chunks, out_path)

    print(f"Chunks generados: {len(chunks)} y guardados en {out_path}")
    print("Ejemplo (primer chunk):")
    print(json.dumps(chunks[0], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
