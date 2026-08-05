import logging

from agents.state import EstadoVenta

logger = logging.getLogger(__name__)

SALUDO_PLANTILLA_DEFAULT = "¡Buenas! 🌱 Bienvenido/a. Soy Santi de Plant Hero. Revise nuestro catálogo aquí: https://shorturl.at/3xXfk"
SALUDO_CORTO_DEFAULT = "¡Hola! 🌱"


def _historial_vacio(chat_history: str) -> bool:
    return not (chat_history or "").strip()


def nodo_bienvenida(state: EstadoVenta) -> dict:
    prompts = state.get("prompts", {}) or {}

    if _historial_vacio(state.get("chat_history", "")):
        saludo = prompts.get("saludo_plantilla", SALUDO_PLANTILLA_DEFAULT)
        logger.info("[bienvenida] cliente nuevo -> saludo de plantilla")
    else:
        saludo = prompts.get("saludo_corto", SALUDO_CORTO_DEFAULT)
        logger.info("[bienvenida] con historial -> saludo corto")

    return {"respuesta": saludo, "etapa": "bienvenida"}
