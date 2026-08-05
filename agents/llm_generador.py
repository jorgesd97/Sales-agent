from google import genai
from google.genai import types
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


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
        self.config = types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=2000,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
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
    ) -> str:
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

**Contexto (base de conocimiento):**
{context}

**Instrucciones:**
- NUNCA saludes ni abras con "hola", "buenas", "qué tal" ni similares. El saludo lo maneja el sistema aparte; si tú saludas, el cliente recibe un saludo doble. Entra directo a responder la consulta.
- Responde usando ÚNICAMENTE la información del contexto. No inventes datos que no estén ahí.
- Si el contexto no tiene la información, dilo con honestidad y ofrece consultar.
- Sé claro y conciso. Respuestas cortas, tono cercano, trato de "usted", máximo 1 emoji.

**Mensaje actual del cliente:** {question}

**Tu respuesta:**
"""
        try:
            response = self.client.models.generate_content(
                model=settings.GEMINI_FLASH_MODEL_HIGH,
                contents=prompt,
                config=self.config,
            )
            answer = (response.text or "").strip()
            logger.info(f"[llm_generador] respuesta generada para: '{question[:60]}'")
            return answer
        except Exception as e:
            logger.error(f"[llm_generador] error: {e}")
            return "Disculpe, tuve un problema al procesar su consulta. ¿Podría repetírmela?"
