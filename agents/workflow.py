import logging

from agent_framework import WorkflowBuilder

from agents.messages import PeticionCliente
from agents.nodes.bienvenida import BienvenidaExecutor
from agents.nodes.consultivo import ConsultivoExecutor, FALLBACK_MSG

logger = logging.getLogger(__name__)


class SalesWorkflow:
    """Reemplaza a `SalesGraph` (LangGraph).

    Equivalencias: `set_entry_point` -> `start_executor=`; `add_edge("a","b")` ->
    `add_edge(a, b)` con objetos; `compile()` -> `build()`; `ainvoke()` -> `run()`.
    No hace falta declarar END: `yield_output()` cierra el workflow.

    Al construir, Agent Framework advierte "Dead-end executors detected:
    ['consultivo']". Es esperado y correcto: consultivo es terminal.
    """

    def __init__(self):
        bienvenida = BienvenidaExecutor()
        consultivo = ConsultivoExecutor()
        self.workflow = (
            WorkflowBuilder(start_executor=bienvenida)
            .add_edge(bienvenida, consultivo)
            .build()
        )

    async def run(
        self,
        question: str = "",
        current_time: str = "",
        prompts: dict = None,
        naturaleza_producto: str = "FISICO",
        simbolo_moneda: str = "S/",
        chat_history: str = "",
    ) -> dict:
        """Misma firma que `SalesGraph.run()` para intercambiabilidad."""
        peticion = PeticionCliente(
            question=question,
            chat_history=chat_history,
            current_time=current_time,
            prompts=prompts or {},
            naturaleza_producto=naturaleza_producto,
            simbolo_moneda=simbolo_moneda,
        )

        resultado = await self.workflow.run(peticion)
        outputs = resultado.get_outputs()

        if not outputs:
            logger.error("[workflow] el workflow no produjo salida")
            return {"answer": FALLBACK_MSG, "is_fallback": True}

        answer = outputs[0]
        return {"answer": answer, "is_fallback": answer == FALLBACK_MSG}
