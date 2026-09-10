import asyncio
import logging

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import OpenAI

from config.settings import settings

logger = logging.getLogger(__name__)


class AzureSearchRetriever:
    """Búsqueda híbrida nativa sobre Azure AI Search.

    Reemplaza la Edge Function `hybrid-search` de Supabase: se manda `search_text`
    (BM25) y `vector_queries` en la MISMA llamada y AI Search fusiona los dos
    rankings con RRF automáticamente.

    Misma interfaz que el `SupabaseRetriever` original (`search`, `format_context`).
    """

    def __init__(self):
        self.client = SearchClient(
            endpoint=settings.AZURE_SEARCH_ENDPOINT,
            index_name=settings.AZURE_SEARCH_INDEX_NAME,
            credential=AzureKeyCredential(settings.AZURE_SEARCH_ADMIN_KEY),
        )
        endpoint = settings.AZURE_OPENAI_ENDPOINT.rstrip("/")
        self.openai_client = OpenAI(
            base_url=f"{endpoint}/openai/v1/",
            api_key=settings.AZURE_OPENAI_API_KEY,
        )

    def _embed(self, texto: str) -> list[float]:
        r = self.openai_client.embeddings.create(
            model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            input=texto,
        )
        return r.data[0].embedding

    def _search_sync(self, query: str, match_count: int) -> list[dict]:
        vq = VectorizedQuery(
            vector=self._embed(query),
            k_nearest_neighbors=match_count,
            fields="content_vector",
        )
        results = self.client.search(
            search_text=query,
            vector_queries=[vq],
            top=match_count,
        )
        return [{"content": r["content"], "title": r["title"]} for r in results]

    async def search(self, query: str, match_count: int = None) -> list[dict]:
        match_count = match_count or settings.DEFAULT_MATCH_COUNT
        try:
            # El SDK de AI Search y el de embeddings son síncronos: se van a un
            # hilo para no bloquear el event loop de FastAPI.
            docs = await asyncio.to_thread(self._search_sync, query, match_count)
            logger.info(f"[hibrida] {len(docs)} chunks para: '{query}'")
            return docs
        except Exception as e:
            logger.error(f"Retriever error: {e}")
            return []

    def format_context(self, documents: list[dict]) -> str:
        return "\n\n".join(d["content"] for d in documents)
