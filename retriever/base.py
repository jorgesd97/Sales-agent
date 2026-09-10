"""Interfaces de los proveedores de datos del retriever.

`PromotionsService` sigue apuntando a Supabase (su migración es una fase
posterior). Aislarlo detrás de este Protocol hace que cambiar el backend sea
sustituir la implementación, sin tocar los executors.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class PromotionsProvider(Protocol):
    async def get_active_promotions(self) -> list[dict]: ...

    def format_promotions(self, promos: list[dict]) -> str: ...


@runtime_checkable
class KnowledgeRetriever(Protocol):
    async def search(self, query: str, match_count: int = None) -> list[dict]: ...

    def format_context(self, documents: list[dict]) -> str: ...
