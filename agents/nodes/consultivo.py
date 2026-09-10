import asyncio
import logging

from agent_framework import Executor, WorkflowContext, handler

from agents.messages import PeticionConSaludo
from agents.sales_agent import crear_agente
from retriever.promotions_service import PromotionsService

logger = logging.getLogger(__name__)

FALLBACK_MSG = "Disculpe, tuve un problema al procesar su consulta. ¿Podría repetírmela?"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1.5

_promotions_service = PromotionsService()


def _construir_prompt_negocio(
    prompt_base: str,
    prompt_nodo: str,
    current_time: str,
    chat_history: str,
    promociones_vigentes: str,
    naturaleza_producto: str,
    simbolo_moneda: str,
) -> str:
    """Mismo prompt de negocio que armaba `LLMGenerador.generar`.

    Ya no lleva el bloque de contexto de la KB (ahora lo trae la tool
    `buscar_en_catalogo`) ni las reglas de formato (las concatena `crear_agente`).
    """
    return f"""{prompt_base}

{prompt_nodo}

=== CONFIGURACIÓN DEL NEGOCIO ===
- Naturaleza del producto: {naturaleza_producto}
- Símbolo de moneda: {simbolo_moneda}

=== FECHA Y HORA ACTUAL (Lima, Perú) ===
{current_time}
Usa SIEMPRE esta fecha/hora como referencia para cualquier cálculo temporal (si un horario ya
pasó, cuándo es "hoy"/"mañana", etc.). Para saber si un horario de HOY ya pasó, compara SOLO la
hora: un horario de hoy ya pasó únicamente si su hora es MENOR que la hora actual. Ejemplo: si
son las 03:19 y el cliente pide 10:00, las 10:00 NO han pasado (10 > 3). Calcula internamente,
nunca narres el cálculo.

**Historial de conversación:**
{chat_history}

**Objetivo:**
Desbes concretar una venta completa y seguir el flujo de venta.
### Flujo de venta
1) === PROMOCIONES DISPONIBLES ===
{promociones_vigentes}

INSTRUCCIONES DE USO DE PROMOCIONES:
- Sé un vendedor PROACTIVO, no reactivo. En cuanto detectes que una promoción es aplicable, anúnciala inmediatamente en la MISMA respuesta donde se genera la oportunidad — no esperes al resumen final ni a que el cliente pregunte.
- Antes de anunciarla, verificá que se cumpla la "condición de aplicación". Si aplica: dispará. Si no aplica: no ofrezcas nada.
- Momento correcto para disparar: cuando el cliente cruza el umbral de aplicación (ejemplo: confirma su segundo sustrato, se acerca a S/120, muestra dudas). Ese es el turno para mencionar la promo, no el turno siguiente.
- No inventes promociones fuera de la lista. Solo podés ofrecer las listadas.
- Usá la "Frase sugerida" como base: adaptá el tono, no cambies números ni condiciones.
- Si la frase tiene un placeholder tipo {{X}}, reemplazalo por el cálculo correcto.
- Si dos promos aplican al mismo tiempo, priorizá la que aparece primero en la lista (mayor prioridad).
- El objetivo del descuento es CERRAR LA VENTA o INCREMENTAR EL TICKET, no regalarlo. Si aplica correctamente, dispará sin miedo.
2) Debes siempre entregar un resumen ordenado de su orden y el costo de envio con un total de la compra.
3) Debes poder tener estos 3 datos para proceder con el pago: a) Pedido b) dirección exacta con referencia y c) Fecha de entrega deseada con una hora o rango horario.
4) Cuando completes el paso 3) debes brindar los medios de pago al cliente usando el query "medios de pago"
5) Al completar la venta agradece al cliente e indicale que se comunicarán con él cuando su pedido salga a entrega.

Toda la información del negocio (productos, precios, stock, envíos, medios de pago)
la obtienes con la herramienta `buscar_en_catalogo`. Si el cliente pregunta algo
concreto del negocio, úsala antes de responder.
"""


class ConsultivoExecutor(Executor):
    """Executor 2 — Responde la consulta con el agente + la tool de catálogo."""

    def __init__(self):
        super().__init__(id="consultivo")

    @handler
    async def run(
        self,
        entrada: PeticionConSaludo,
        ctx: WorkflowContext[None, str],   # [no envía mensajes, produce str]
    ) -> None:
        peticion = entrada.peticion
        prompts = peticion.prompts or {}

        try:
            promos = await _promotions_service.get_active_promotions()
            promociones_vigentes = _promotions_service.format_promotions(promos)

            prompt_negocio = _construir_prompt_negocio(
                prompt_base=prompts.get("base", ""),
                prompt_nodo=prompts.get("consultivo", ""),
                current_time=peticion.current_time,
                chat_history=peticion.chat_history,
                promociones_vigentes=promociones_vigentes,
                naturaleza_producto=peticion.naturaleza_producto,
                simbolo_moneda=peticion.simbolo_moneda,
            )

            texto = await self._ejecutar_agente(prompt_negocio, peticion.question)

        except Exception as e:
            logger.error(f"[consultivo] error fatal: {e}")
            await ctx.yield_output(FALLBACK_MSG)
            return

        saludo = (entrada.saludo_prefijo or "").strip()
        respuesta = f"{saludo}\n---\n{texto}" if saludo else texto
        await ctx.yield_output(respuesta)

    async def _ejecutar_agente(self, prompt_negocio: str, question: str) -> str:
        """Reintenta ante rate limit, igual que hacía `LLMGenerador.generar`."""
        agente = crear_agente(prompt_negocio)

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resultado = await agente.run(question)
                texto = (resultado.text or "").strip()
                if not texto:
                    logger.warning(f"[consultivo] respuesta vacía en intento {attempt}")
                    raise ValueError("empty_response")

                logger.info(
                    f"[consultivo] respuesta generada para: '{question[:60]}' (intento {attempt})"
                )
                return texto

            except Exception as e:
                last_error = e
                error_str = str(e)
                is_rate_limit = "429" in error_str or "rate limit" in error_str.lower()

                if is_rate_limit and attempt < MAX_RETRIES:
                    logger.warning(
                        f"[consultivo] 429 rate limit en intento {attempt}/{MAX_RETRIES}, "
                        f"reintentando en {RETRY_DELAY_SECONDS}s..."
                    )
                    await asyncio.sleep(RETRY_DELAY_SECONDS)
                    continue

                logger.error(f"[consultivo] error en intento {attempt}: {e}")
                break

        raise RuntimeError(f"agente falló tras {MAX_RETRIES} intentos: {last_error}")
