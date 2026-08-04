from langgraph.graph import StateGraph, END
from typing import TypedDict
import logging

from .research_agent import ResearchAgent
from .router_agent import RouterAgent
from retriever.supabase_retriever import SupabaseRetriever

logger = logging.getLogger(__name__)

RETRIEVE_QUERIES = {
    "bienvenida": None,
    "objeciones": None,
    "logistica": "políticas de despacho envío costo",
    "confirmacion_datos": None,
    "pago": "métodos de pago",
    "validacion": "métodos de pago validación",
    "agregar_producto": None,
}

PROMPT_KEYS = {
    "bienvenida": "bienvenida",
    "objeciones": "objeciones",
    "logistica": "logistica",
    "confirmacion_datos": "datos",
    "pago": "pago",
    "validacion": "validacion",
    "agregar_producto": "agregar",
}


class AgentState(TypedDict):
    question: str
    current_time: str
    chat_history: str
    context: str
    draft_answer: str
    prompts: dict
    naturaleza_producto: str
    simbolo_moneda: str
    table_name: str
    etapa: str


class AgentWorkflow:
    def __init__(self):
        self.researcher = ResearchAgent()
        self.router = RouterAgent()
        self.retriever = SupabaseRetriever()
        self.compiled = self._build()

    def _build(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("enrutador", self._router_step)
        workflow.add_node("bienvenida", self._make_stage_step("bienvenida"))
        workflow.add_node("objeciones", self._make_stage_step("objeciones"))
        workflow.add_node("logistica", self._make_stage_step("logistica"))
        workflow.add_node("confirmacion_datos", self._make_stage_step("confirmacion_datos"))
        workflow.add_node("guardian", self._guardian_step)
        workflow.add_node("pago", self._make_stage_step("pago"))
        workflow.add_node("validacion", self._make_stage_step("validacion"))
        workflow.add_node("agregar_producto", self._make_stage_step("agregar_producto"))

        workflow.set_entry_point("enrutador")

        workflow.add_conditional_edges(
            "enrutador",
            self._route_to_stage,
            {
                "bienvenida": "bienvenida",
                "objeciones": "objeciones",
                "logistica": "logistica",
                "confirmacion_datos": "confirmacion_datos",
                "pago": "guardian",
                "validacion": "validacion",
                "agregar_producto": "agregar_producto",
            },
        )

        workflow.add_edge("bienvenida", END)
        workflow.add_edge("objeciones", END)
        workflow.add_edge("logistica", END)
        workflow.add_edge("confirmacion_datos", END)
        workflow.add_edge("pago", END)
        workflow.add_edge("validacion", END)
        workflow.add_edge("agregar_producto", END)

        workflow.add_conditional_edges(
            "guardian",
            self._guardian_decision,
            {"pago": "pago", "confirmacion_datos": "confirmacion_datos"},
        )

        return workflow.compile()

    async def _router_step(self, state: AgentState) -> dict:
        result = await self.router.classify_stage(
            chat_history=state.get("chat_history", ""),
            question=state["question"],
            naturaleza_producto=state.get("naturaleza_producto", "FISICO"),
        )
        etapa = result.get("etapa", "bienvenida")
        logger.info(f"Router decided stage: {etapa}")
        return {"etapa": etapa}

    def _route_to_stage(self, state: AgentState) -> str:
        return state.get("etapa", "bienvenida")

    def _guardian_step(self, state: AgentState) -> dict:
        return {}

    def _guardian_decision(self, state: AgentState) -> str:
        chat = state.get("chat_history", "").lower()
        has_nombre = any(kw in chat for kw in ["nombre:", "nombre completo:", "me llamo", "mi nombre es", "soy "])
        has_direccion = any(kw in chat for kw in ["dirección:", "direccion:", "av.", "av ", "calle ", "jr.", "jr ", "mz ", "lt "])

        naturaleza = state.get("naturaleza_producto", "FISICO")
        if naturaleza == "DIGITAL":
            if has_nombre:
                logger.info("Guardian: data complete (DIGITAL), proceeding to pago")
                return "pago"
        else:
            has_horario = any(kw in chat for kw in ["horario:", "hora:", "am", "pm", "de la mañana", "de la tarde"])
            if has_nombre and has_direccion and has_horario:
                logger.info("Guardian: data complete (FISICO), proceeding to pago")
                return "pago"

        logger.info("Guardian: data incomplete, routing back to confirmacion_datos")
        return "confirmacion_datos"

    def _make_stage_step(self, stage_name: str):
        async def stage_step(state: AgentState) -> dict:
            query = RETRIEVE_QUERIES.get(stage_name)
            if query is None:
                query = state["question"]
            if stage_name == "objeciones":
                query = f"{state['question']} objeciones descuentos"

            documents = await self.retriever.search(
                query=query,
                table_name=state.get("table_name", "kb_demo"),
                match_count=5,
            )
            context = self.retriever.format_context(documents)
            logger.info(f"[{stage_name}] Retrieved {len(documents)} chunks ({len(context)} chars)")

            prompts = state.get("prompts", {})
            prompt_base = prompts.get("base", "")
            prompt_key = PROMPT_KEYS.get(stage_name, stage_name)
            prompt_nodo = prompts.get(prompt_key, "")

            answer = await self.researcher.generate(
                question=state["question"],
                current_time=state.get("current_time", ""),
                context=context,
                prompt_base=prompt_base,
                prompt_nodo=prompt_nodo,
                chat_history=state.get("chat_history", ""),
            )
            logger.info(f"[{stage_name}] Answer: {answer[:150]}...")
            return {"draft_answer": answer, "context": context}

        return stage_step

    async def run(
        self,
        question: str,
        current_time: str = "",
        prompts: dict = None,
        naturaleza_producto: str = "FISICO",
        simbolo_moneda: str = "S/",
        chat_history: str = "",
        table_name: str = "kb_demo",
    ) -> dict:
        initial_state = AgentState(
            question=question,
            current_time=current_time,
            chat_history=chat_history,
            context="",
            draft_answer="",
            prompts=prompts or {},
            naturaleza_producto=naturaleza_producto,
            simbolo_moneda=simbolo_moneda,
            table_name=table_name,
            etapa="",
        )
        final_state = await self.compiled.ainvoke(initial_state)
        return {"answer": final_state["draft_answer"]}
