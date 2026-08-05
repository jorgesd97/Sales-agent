from langgraph.graph import StateGraph, END

from agents.state import EstadoVenta
from agents.nodes.bienvenida import nodo_bienvenida


class SalesGraph:
    def __init__(self):
        self.compiled = self._build()

    def _build(self):
        g = StateGraph(EstadoVenta)
        g.add_node("bienvenida", nodo_bienvenida)
        g.set_entry_point("bienvenida")
        g.add_edge("bienvenida", END)
        return g.compile()

    async def run(
        self,
        question: str = "",
        current_time: str = "",
        prompts: dict = None,
        naturaleza_producto: str = "FISICO",
        simbolo_moneda: str = "S/",
        chat_history: str = "",
        table_name: str = "kb_demo",
    ) -> dict:
        estado_inicial: EstadoVenta = {
            "question": question,
            "chat_history": chat_history,
            "current_time": current_time,
            "prompts": prompts or {},
            "naturaleza_producto": naturaleza_producto,
            "simbolo_moneda": simbolo_moneda,
            "table_name": table_name,
            "respuesta": "",
            "etapa": "",
        }
        final = await self.compiled.ainvoke(estado_inicial)
        return {"answer": final["respuesta"]}
