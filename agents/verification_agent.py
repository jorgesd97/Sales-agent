from google import genai
from google.genai import types
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from config.settings import settings

logger = logging.getLogger(__name__)


class VerificationAgent:
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
            temperature=0,
            max_output_tokens=300,
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

    async def check(self, answer: str, question: str, chat_history: str, sales_flow: str) -> dict:
        lima_tz = ZoneInfo("America/Lima")
        now_lima = datetime.now(lima_tz)
        dias = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
        meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
                "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
        dia_semana = dias[now_lima.weekday()]
        mes = meses[now_lima.month - 1]
        current_time_str = f"{dia_semana} {now_lima.day} de {mes} de {now_lima.year}, {now_lima.strftime('%H:%M')} horas (formato 24h)"
        conversacion = f"{chat_history}\n[CLIENTE]: {question}"

        prompt = f"""[Current date and time in Lima, Peru: {current_time_str}]

You are a sales flow verifier. Below is a sales conversation in chronological order between a CUSTOMER and the SELLER (an AI assistant). After the conversation there is a PROPOSED SELLER RESPONSE that you must evaluate.

In the conversation, "Human" or "[CLIENTE]" means the CUSTOMER, and "AI" means the SELLER. The PROPOSED SELLER RESPONSE is the seller's reply to the LAST customer message in the conversation.

**Sales Flow (the steps and non-negotiable rules the response MUST respect):**
{sales_flow}

**CONVERSATION (chronological order):**
{conversacion}

**PROPOSED SELLER RESPONSE (evaluate THIS):**
{answer}

**Check the following about the PROPOSED SELLER RESPONSE:**
1. Does it respect the sales flow (does not skip mandatory steps)?
2. Does it ask for information the customer has ALREADY provided anywhere in the conversation above?
3. Are there temporal inconsistencies against the current date/time given above? (offering a time already passed, treating "tomorrow" as "today", etc.)
4. Does it repeat information textually identical to a previous SELLER turn?

If the SELLER previously asked something (e.g. "would you like to see the image?") and the CUSTOMER's last message confirms it (e.g. "sí", "si pf", "dale", "ok"), then the seller acting on that confirmation is CORRECT flow, not an error.

**Respond ONLY with the following format:**
Flujo_correcto: YES/NO
Problemas: [brief list of detected problems, or "ninguno"]
"""
        try:
            response = self.client.models.generate_content(
                model=settings.GEMINI_FLASH_MODEL_LOW,
                contents=prompt,
                config=self.config
            )

            if not response.text:
                logger.warning("Verification returned empty response, fail-open")
                return {"verification_report": "Empty response from verifier", "is_valid": True}

            report = response.text.strip()
            logger.info(f"Verification report: {report}")

            parsed = self._parse_report(report)
            return {
                "verification_report": report,
                "is_valid": "YES" in parsed.get("flujo_correcto", ""),
            }

        except Exception as e:
            logger.error(f"Verification error: {e}")
            return {"verification_report": f"Verification error: {e}", "is_valid": True}

    def _parse_report(self, report: str) -> dict:
        parsed = {}
        for line in report.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip().upper()
                if key == "flujo_correcto":
                    parsed["flujo_correcto"] = value
        return parsed