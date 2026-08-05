import logging

from agents.state import EstadoVenta

logger = logging.getLogger(__name__)

# Fallback si n8n no envía la plantilla en el campo prompts.
SALUDO_PLANTILLA_DEFAULT = "¡Buenas! 🌱 Bienvenido/a. Soy Santi de Plant Hero. Revise nuestro catálogo aquí: https://shorturl.at/3xXfk"


def _historial_vacio(chat_history: str) -> bool:
    return not (chat_history or "").strip()


def nodo_bienvenida(state: EstadoVenta) -> dict:
    """Nodo 1 — Saludo. Decisión determinista, sin LLM.

    Ya NO produce la respuesta final: produce solo un PREFIJO de saludo que el
    nodo consultivo antepone a su respuesta en el mismo turno.

    Sin historial -> cliente nuevo -> prefijo con saludo de plantilla.
    Con historial -> prefijo vacío (no se vuelve a saludar).
    """
    prompts = state.get("prompts", {}) or {}

    if _historial_vacio(state.get("chat_history", "")):
        saludo = prompts.get("saludo_plantilla", SALUDO_PLANTILLA_DEFAULT)
        logger.info("[bienvenida] cliente nuevo -> prefijo de saludo")
    else:
        saludo = ""
        logger.info("[bienvenida] con historial -> sin saludo")

    return {"saludo_prefijo": saludo, "etapa": "consultivo"}
