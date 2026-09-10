import logging

from agent_framework import Executor, WorkflowContext, handler

from agents.messages import PeticionCliente, PeticionConSaludo

logger = logging.getLogger(__name__)

# Fallback si n8n no envía la plantilla en el campo prompts.
SALUDO_PLANTILLA_DEFAULT = "¡Buenas! 🌱 Bienvenido/a. Soy Santi de Plant Hero. Revise nuestro catálogo aquí: https://shorturl.at/3xXfk"


def _historial_vacio(chat_history: str) -> bool:
    return not (chat_history or "").strip()


class BienvenidaExecutor(Executor):
    """Executor 1 — Saludo. Decisión determinista, sin LLM.

    No produce la respuesta final: produce solo un PREFIJO de saludo que el
    executor consultivo antepone a su respuesta en el mismo turno.

    Sin historial -> cliente nuevo -> prefijo con saludo de plantilla.
    Con historial -> prefijo vacío (no se vuelve a saludar).
    """

    def __init__(self):
        super().__init__(id="bienvenida")

    @handler
    async def run(
        self,
        peticion: PeticionCliente,
        ctx: WorkflowContext[PeticionConSaludo],
    ) -> None:
        prompts = peticion.prompts or {}

        if _historial_vacio(peticion.chat_history):
            saludo = prompts.get("saludo_plantilla", SALUDO_PLANTILLA_DEFAULT)
            logger.info("[bienvenida] cliente nuevo -> prefijo de saludo")
        else:
            saludo = ""
            logger.info("[bienvenida] con historial -> sin saludo")

        await ctx.send_message(
            PeticionConSaludo(peticion=peticion, saludo_prefijo=saludo)
        )
