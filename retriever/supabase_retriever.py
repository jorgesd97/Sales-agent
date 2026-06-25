import httpx
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


class SupabaseRetriever:
    def __init__(self):
        self.url = settings.SUPABASE_EDGE_FUNCTION_URL
        self.headers = {
            "Authorization": f"Bearer {settings.SUPABASE_ANON_KEY}",
            "Content-Type": "application/json",
        }

    async def search(
        self,
        query: str,
        table_name: str = None,
        match_count: int = None,
    ) -> list[dict]:
        table_name = table_name or settings.DEFAULT_TABLE_NAME
        match_count = match_count or settings.DEFAULT_MATCH_COUNT

        payload = {
            "query": query,
            "table_name": table_name,
            "match_count": match_count,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.url,
                    headers=self.headers,
                    json=payload,
                )
                response.raise_for_status()
                documents = response.json()

            logger.info(f"Retrieved {len(documents)} chunks for: '{query}'")
            return documents

        except Exception as e:
            logger.error(f"Retriever error: {e}")
            return []

    def format_context(self, documents: list[dict]) -> str:
        return "\n\n".join([doc["content"] for doc in documents])