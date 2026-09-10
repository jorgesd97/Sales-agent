"""Mensajes tipados que viajan por el workflow.

Reemplaza el `EstadoVenta` (TypedDict compartido) de LangGraph: Agent Framework
rutea **por tipo**, así que cada executor declara en su firma qué consume y qué
produce en vez de fusionar parcialmente un estado global.
"""

from dataclasses import dataclass, field


@dataclass
class PeticionCliente:
    """Entrada del workflow. Equivale al estado inicial de LangGraph."""

    question: str
    chat_history: str = ""
    current_time: str = ""
    prompts: dict = field(default_factory=dict)
    naturaleza_producto: str = "FISICO"
    simbolo_moneda: str = "S/"


@dataclass
class PeticionConSaludo:
    """Producido por bienvenida, consumido por consultivo."""

    peticion: PeticionCliente
    saludo_prefijo: str
