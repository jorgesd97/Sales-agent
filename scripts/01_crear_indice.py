"""Crea (o actualiza) el índice de Azure AI Search. Idempotente.

    python scripts/01_crear_indice.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)

from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# text-embedding-3-small
EMBEDDING_DIMENSIONS = 1536


def build_index() -> SearchIndex:
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        # filterable pensando en multi-tenancy y filtros futuros
        SimpleField(name="business_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(
            name="category",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="source", type=SearchFieldDataType.String),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            hidden=True,
            vector_search_dimensions=EMBEDDING_DIMENSIONS,
            vector_search_profile_name="perfil-hnsw",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="algo-hnsw",
                parameters=HnswParameters(metric=VectorSearchAlgorithmMetric.COSINE),
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="perfil-hnsw",
                algorithm_configuration_name="algo-hnsw",
            )
        ],
    )

    return SearchIndex(
        name=settings.AZURE_SEARCH_INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
    )


def main():
    client = SearchIndexClient(
        endpoint=settings.AZURE_SEARCH_ENDPOINT,
        credential=AzureKeyCredential(settings.AZURE_SEARCH_ADMIN_KEY),
    )
    # create_or_update -> idempotente: se puede correr las veces que haga falta.
    result = client.create_or_update_index(build_index())
    logger.info(f"Índice '{result.name}' creado/actualizado con {len(result.fields)} campos")


if __name__ == "__main__":
    main()
