"""Trocea un markdown de la base de conocimiento, genera embeddings y lo sube
al índice de Azure AI Search. Idempotente: el `id` es determinístico y
`upload_documents` hace upsert.

    python scripts/02_cargar_kb.py kb/dodo.md --business-id dodo --category catalogo

El troceo es por encabezado `###`, igual que hace el pipeline de n8n con Docling.
"""

import argparse
import hashlib
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import OpenAI

from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BATCH_SIZE = 50


def trocear(texto: str) -> list[dict]:
    """Parte el markdown en secciones que arrancan en un encabezado `###`.

    Lo anterior al primer `###` se guarda como sección "Introducción".
    """
    partes = re.split(r"^###\s+(.+)$", texto, flags=re.MULTILINE)

    chunks = []
    preambulo = partes[0].strip()
    if preambulo:
        chunks.append({"title": "Introducción", "content": preambulo})

    # partes queda como [preambulo, titulo1, cuerpo1, titulo2, cuerpo2, ...]
    for titulo, cuerpo in zip(partes[1::2], partes[2::2]):
        cuerpo = cuerpo.strip()
        if not cuerpo:
            continue
        chunks.append({"title": titulo.strip(), "content": f"### {titulo.strip()}\n{cuerpo}"})

    return chunks


def doc_id(source: str, title: str) -> str:
    """id determinístico -> volver a correr el script actualiza, no duplica."""
    return hashlib.sha1(f"{source}::{title}".encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archivo", help="Markdown de la base de conocimiento")
    parser.add_argument("--business-id", default="default")
    parser.add_argument("--category", default="general")
    args = parser.parse_args()

    ruta = Path(args.archivo)
    texto = ruta.read_text(encoding="utf-8")
    chunks = trocear(texto)
    logger.info(f"{len(chunks)} chunks obtenidos de {ruta.name}")

    endpoint = settings.AZURE_OPENAI_ENDPOINT.rstrip("/")
    openai_client = OpenAI(
        base_url=f"{endpoint}/openai/v1/",
        api_key=settings.AZURE_OPENAI_API_KEY,
    )
    search_client = SearchClient(
        endpoint=settings.AZURE_SEARCH_ENDPOINT,
        index_name=settings.AZURE_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(settings.AZURE_SEARCH_ADMIN_KEY),
    )

    documentos = []
    for chunk in chunks:
        embedding = openai_client.embeddings.create(
            model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            input=chunk["content"],
        ).data[0].embedding

        documentos.append(
            {
                "id": doc_id(ruta.name, chunk["title"]),
                "business_id": args.business_id,
                "category": args.category,
                "title": chunk["title"],
                "content": chunk["content"],
                "source": ruta.name,
                "content_vector": embedding,
            }
        )

    for i in range(0, len(documentos), BATCH_SIZE):
        lote = documentos[i : i + BATCH_SIZE]
        resultados = search_client.upload_documents(documents=lote)
        fallidos = [r for r in resultados if not r.succeeded]
        if fallidos:
            logger.error(f"{len(fallidos)} documentos fallaron: {[f.key for f in fallidos]}")
        logger.info(f"Subidos {i + len(lote)}/{len(documentos)}")


if __name__ == "__main__":
    main()
