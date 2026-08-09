import asyncio

from google import genai
from google.genai import types
import logging

from config.settings import settings
from retriever.supabase_retriever import SupabaseRetriever
from retriever.promotions_service import PromotionsService

logger = logging.getLogger(__name__)

FALLBACK_MSG = "Disculpe, tuve un problema al procesar su consulta. ¿Podría repetírmela?"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1.5


class LLMGenerador:
    """Ayudante reutilizable para generar texto con Gemini (Vertex AI).
    No es un nodo del grafo: los nodos lo invocan para redactar su respuesta.
    """

    def __init__(self):
        import os
        from google.oauth2 import service_account

        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        credentials = service_account.Credentials.from_service_account_file(
            creds_path, scopes=scopes
        )
        self.client = genai.Client(
            vertexai=True,
            project=settings.GOOGLE_CLOUD_PROJECT,
            location=settings.GOOGLE_CLOUD_LOCATION,
            credentials=credentials,
        )
        self.retriever = SupabaseRetriever()
        self.promotions_service = PromotionsService()
        self.config = types.GenerateContentConfig(
            temperature=0.4,
            max_output_tokens=2000,
            thinking_config=types.ThinkingConfig(thinking_budget=512),
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            ],
        )

    async def generar(
        self,
        question: str,
        current_time: str = "",
        context: str = "",
        prompt_base: str = "",
        prompt_nodo: str = "",
        chat_history: str = "",
        table_name: str = "kb_demo",
    ) -> tuple[str, bool]:
        promos = await self.promotions_service.get_active_promotions()
        promociones_vigentes = self.promotions_service.format_promotions(promos)

        prompt = f"""{prompt_base}

{prompt_nodo}

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
3) Debes poder tener estos 3 datos para proceder con el pago: a) Pedido b) dirección exacta con referencia y c) Fecha de entrega deseada
4) Cuando completes el paso 3) debes brindar los medios de pago al cliente usando el query "medios de pago"
5) Al completar la venta agradece al cliente e indicale que se comunicarán con él cuando su pedido salga a entrega.
**Contexto (base de conocimiento):**
{context}

**Instrucciones:**
- NUNCA saludes ni abras con "hola", "buenas", "qué tal" ni similares. El saludo lo maneja el sistema aparte; si tú saludas, el cliente recibe un saludo doble. Entra directo a responder la consulta.
- Si el contexto no tiene la información, dilo con honestidad y ofrece consultar.
- Sé claro y conciso. Respuestas cortas, tono cercano, trato de "usted", máximo 1 emoji.
- FORMATO PARA WHATSAPP: escribe como se chatea en el celular, en mensajes cortos. Si tu respuesta es algo largo (por ejemplo, responder sobre un producto Y sobre otro tema), sepáralas en globos usando una línea con exactamente tres guiones "---" entre ellas. Cada globo debe ser breve (2 a 3 líneas). Usa "---" solo entre ideas separadas; no lo pongas al inicio, ni al final, ni dentro de una misma idea. Si la respuesta es una sola idea corta, no uses "---".

**Mensaje actual del cliente:** {question}

**Tu respuesta:**
"""
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.models.generate_content(
                    model=settings.GEMINI_FLASH_MODEL_HIGH,
                    contents=prompt,
                    config=self.config,
                )
                answer = (response.text or "").strip()
                if not answer:
                    logger.warning(f"[llm_generador] respuesta vacía en intento {attempt}")
                    raise ValueError("empty_response")

                logger.info(f"[llm_generador] respuesta generada para: '{question[:60]}' (intento {attempt})")
                return answer, False

            except Exception as e:
                last_error = e
                error_str = str(e)
                is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str

                if is_rate_limit and attempt < MAX_RETRIES:
                    logger.warning(
                        f"[llm_generador] 429 rate limit en intento {attempt}/{MAX_RETRIES}, "
                        f"reintentando en {RETRY_DELAY_SECONDS}s..."
                    )
                    await asyncio.sleep(RETRY_DELAY_SECONDS)
                    continue

                logger.error(f"[llm_generador] error en intento {attempt}: {e}")
                break

        logger.error(f"[llm_generador] retries agotados o error fatal, último error: {last_error}")
        return FALLBACK_MSG, True
