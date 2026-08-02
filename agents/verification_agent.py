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

    async def check(self, answer: str, chat_history: str, system_prompt: str, context: str = "") -> dict:
        context_section = ""
        lima_tz = ZoneInfo("America/Lima")
        now_lima = datetime.now(lima_tz)
        dias = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
        meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", 
                "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
        dia_semana = dias[now_lima.weekday()]
        mes = meses[now_lima.month - 1]
        current_time_str = f"{dia_semana} {now_lima.day} de {mes} de {now_lima.year}, {now_lima.strftime('%H:%M')} horas (formato 24h)"
        if context:
            context_section = f"""
**Knowledge Base (to verify product/price data):**
{context}
"""
        prompt = f"""[Current date and time in Lima, Peru: {current_time_str}]

You are a sales flow verifier. Your job is to check whether a proposed response follows the sales flow correctly given the conversation history.

**Check the following:**
1. Does the response respect the sales flow defined in the system prompt (does not skip mandatory steps)?
2. Does the response ask for information the customer has ALREADY provided in the chat history? (error)
3. Are there any temporal inconsistencies? Use ONLY the current date/time provided above as your reference. Do NOT recalculate the day of the week yourself — trust the date given. Flag a problem only if the response clearly contradicts the provided current date (e.g. offering a time that has already passed, or stating a wrong "tomorrow" relative to the given date).
4. Does the response repeat obsolete information, or copy an earlier turn word-for-word?
5. Does the response invent products or prices not found in the knowledge base (if provided)?

**Severity calibration (IMPORTANT):**
Mark "NO" ONLY when there is a CLEAR, CONCRETE error that would harm the sale or confuse the customer — for example: requesting payment before collecting the required data, asking for a datum the customer already gave, or a real temporal contradiction against the current date provided above.
Do NOT mark "NO" for stylistic issues, tone, "incomplete" benefit descriptions, missing emojis, or minor wording. If the response is useful and correct for the customer, mark "YES". When in doubt, mark "YES".

**System Prompt (contains the expected sales flow):**
{system_prompt}

**Chat History:**
{chat_history}

**Proposed Response:**
{answer}
{context_section}
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
                "is_valid": parsed.get("flujo_correcto") == "YES",
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