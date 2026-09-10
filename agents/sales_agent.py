import logging
from typing import Annotated

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

from config.settings import settings
from retriever.azure_search_retriever import AzureSearchRetriever

logger = logging.getLogger(__name__)

_retriever = AzureSearchRetriever()


async def buscar_en_catalogo(
    query: Annotated[
        str,
        "La consulta de búsqueda. Reformula la pregunta del cliente usando "
        "términos del catálogo. Ej: si pregunta '¿y eso sirve pa mis "
        "orquídeas?', busca 'corteza de pino orquídeas drenaje'.",
    ],
) -> str:
    """Busca información en la base de conocimiento del negocio: productos,
    precios, stock, zonas y costos de envío, métodos de pago, promociones,
    políticas de devolución y preguntas frecuentes.

    Úsala SIEMPRE que el cliente pregunte algo concreto del negocio.
    NO la uses para saludos, agradecimientos o despedidas.
    """
    logger.info(f"[tool] buscar_en_catalogo(query='{query}')")
    docs = await _retriever.search(query=query, match_count=settings.DEFAULT_MATCH_COUNT)
    return _retriever.format_context(docs)


# No negociables: se concatenan SIEMPRE al final del prompt, después de los
# prompts dinámicos que llegan de n8n. Si se pierden, el agente devuelve muros
# de texto que rompen la UX de WhatsApp.
REGLAS_FORMATO = """
**Reglas de formato (obligatorias):**
- NUNCA saludes ni abras con "hola" — el saludo lo maneja el sistema aparte.
- Responde SOLO con lo que devuelva la herramienta. Si no está ahí, dilo con
  honestidad. NUNCA inventes precios, stock ni políticas.
- SÉ BREVE: esto es WhatsApp. Máximo 3 o 4 líneas por bloque.
- PROHIBIDO: listas de más de 3 ítems, pasos numerados, tablas, volcar el
  catálogo completo. Si preguntan por UN producto, menciona ESE producto.
- Trato de "usted". Máximo 1 emoji por respuesta.
- Si tu respuesta cubre más de una idea, sepáralas con una línea que diga
  exactamente "---". No pongas "---" al final ni al inicio del mensaje.
"""


def crear_agente(instrucciones_negocio: str = "") -> Agent:
    """El prompt del negocio (de n8n) va PRIMERO; las reglas de formato al final."""
    client = OpenAIChatClient(
        settings.AZURE_OPENAI_DEPLOYMENT,          # `model` es posicional
        api_key=settings.AZURE_OPENAI_API_KEY,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    )
    return Agent(
        client=client,
        name="SalesAgent",
        instructions=f"{instrucciones_negocio}\n\n{REGLAS_FORMATO}",
        tools=[buscar_en_catalogo],
    )
