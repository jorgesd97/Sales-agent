import logging

from agents.state import EstadoVenta
from agents.llm_generador import LLMGenerador
from retriever.supabase_retriever import SupabaseRetriever

logger = logging.getLogger(__name__)

# Query fija amplia: trae un panorama completo de la KB en cada turno.
QUERY_CONSULTIVA = (
    "productos precios existencia stock envío costos delivery "
    "métodos de pago yape promociones descuentos"
)

_llm = LLMGenerador()
_retriever = SupabaseRetriever()


async def nodo_consultivo(state: EstadoVenta) -> dict:
    documents = await _retriever.search(
        query=state.get("question", ""),
        table_name=state.get("table_name", "kb_demo"),
        match_count=3,
    )
    context = _retriever.format_context(documents)
    logger.info(f"[consultivo] {len(documents)} chunks recuperados ({len(context)} chars)")

    prompts = state.get("prompts", {}) or {}
    respuesta_llm, is_fallback = await _llm.generar(
        question=state.get("question", ""),
        current_time=state.get("current_time", ""),
        context=context,
        prompt_base=prompts.get("base", ""),
        prompt_nodo=prompts.get("consultivo", ""),
        chat_history=state.get("chat_history", ""),
        table_name=state.get("table_name", "kb_demo"),
    )

    if is_fallback:
        return {
            "respuesta": respuesta_llm,
            "context": context,
            "etapa": "consultivo",
            "is_fallback": True,
        }

    saludo = (state.get("saludo_prefijo", "") or "").strip()
    if saludo:
        respuesta = f"{saludo}\n---\n{respuesta_llm}"
    else:
        respuesta = respuesta_llm

    return {
        "respuesta": respuesta,
        "context": context,
        "etapa": "consultivo",
        "is_fallback": False,
    }
