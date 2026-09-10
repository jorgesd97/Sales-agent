"""Interfaz del proveedor de memoria conversacional.

`PostgresMemory` sigue apuntando al Postgres de Supabase (su migración a Cosmos
DB / Azure PostgreSQL es una fase posterior). Aislarlo detrás de este Protocol
hace que cambiar el backend sea sustituir la implementación, sin tocar `main.py`.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class MemoryProvider(Protocol):
    def get_history(
        self, session_id: str, table_name: str = ..., limit: int = ...
    ) -> str: ...

    def save_message(
        self, session_id: str, msg_type: str, content: str, table_name: str = ...
    ) -> None: ...

    def save_interaction(
        self, session_id: str, question: str, answer: str, table_name: str = ...
    ) -> None: ...
