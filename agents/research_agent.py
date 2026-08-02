from google import genai
from google.genai import types
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


class ResearchAgent:
    def __init__(self):
        import os
        from google.oauth2 import service_account
        
        # 1. Cargamos explícitamente el archivo JSON que tu función setup_google_credentials() escribió en el disco
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        credentials = service_account.Credentials.from_service_account_file(
        creds_path, 
        scopes=scopes  # <-- ESTO SOLUCIONA EL "INVALID_SCOPE"
    )
        self.client = genai.Client(
        vertexai=True,
        project=settings.GOOGLE_CLOUD_PROJECT,
        location=settings.GOOGLE_CLOUD_LOCATION,
        credentials=credentials 
    )
        self.config = types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=2000,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_NONE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_NONE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_NONE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_NONE",
                ),
            ]
        )

    async def generate(
        self,
        question: str,
        context: str,
        system_prompt: str = "",
        chat_history: str = "",
        correction_feedback: str = "",
    ) -> str:
        correction_section = ""
        if correction_feedback:
            correction_section = f"""

**IMPORTANTE — Tu respuesta anterior fue rechazada por el verificador de flujo. Motivo:**
{correction_feedback}
Genera una nueva respuesta que corrija ese problema específico. No repitas el mismo error.
"""

        prompt = f"""
{system_prompt}

**Chat History:**
{chat_history}

**Instructions:**
- Answer the following question using ONLY the provided context.
- Be clear, concise, and factual.
- If the context doesn't contain enough information, say so honestly.
- Do NOT invent information that is not in the context.

**Question:** {question}

**Context:**
{context}
{correction_section}
**Provide your answer below:**
"""

        try:
            # 3. Consumimos el modelo desde el cliente unificado
            response = self.client.models.generate_content(
                model=settings.GEMINI_FLASH_MODEL_HIGH,
                contents=prompt,
                config=self.config
            )
            answer = response.text.strip()
            logger.info(f"Research agent generated answer for: '{question}'")
            return answer

        except Exception as e:
            logger.error(f"Research agent error: {e}")
            return "Lo siento, hubo un error al procesar tu consulta."