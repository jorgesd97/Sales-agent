from google import genai
from google.genai import types
import json
import logging
import re

from config.settings import settings

logger = logging.getLogger(__name__)


class SalesClassifier:
    def __init__(self):
        import os
        from google.oauth2 import service_account

        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        credentials = service_account.Credentials.from_service_account_file(
            creds_path,
            scopes=scopes,
        )
        self.client = genai.Client(
            vertexai=True,
            project=settings.GOOGLE_CLOUD_PROJECT,
            location=settings.GOOGLE_CLOUD_LOCATION,
            credentials=credentials,
        )
        self.config = types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=500,
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
            ],
        )

    async def classify(self, chat_history: str, sales_criteria: str) -> dict:
        prompt = f"""You are a sales classification agent. Analyze the following conversation to determine if a sale was completed.

**Your tasks:**
1. Confirm whether there is a validated payment in the conversation (look for the marker [COMPROBANTE_PAGO]).
2. Check each criterion from the sales criteria against the conversation. Be STRICT: mark datos_completos as false if ANY criterion is missing.
3. Extract the sale data from the conversation. If a data point is NOT found in the conversation, set it as null and list it in datos_faltantes. Do NOT invent data.

**Sales Criteria:**
{sales_criteria}

**Complete Conversation:**
{chat_history}

**Respond ONLY with valid JSON, no additional text, no markdown, no ```json fences. Only the JSON object:**
{{
  "venta_cerrada": true/false,
  "datos_completos": true/false,
  "datos_faltantes": ["..."] or null,
  "datos_venta": {{
    "pedido": "..." or null,
    "monto": "..." or null,
    "fecha_entrega": "..." or null,
    "nombre": "..." or null,
    "whatsapp": "..." or null,
    "direccion": "..." or null,
    "fecha_voucher": "..." or null
  }}
}}
"""

        try:
            response = self.client.models.generate_content(
                model=settings.GEMINI_FLASH_MODEL_LOW,
                contents=prompt,
                config=self.config,
            )

            if not response.text:
                logger.warning("Sales classifier returned empty response")
                return self._safe_fallback("empty_response")

            raw = response.text.strip()
            raw = re.sub(r"^```json\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            result = json.loads(raw)
            logger.info(f"Sales classification: venta_cerrada={result.get('venta_cerrada')}, datos_completos={result.get('datos_completos')}")
            return result

        except json.JSONDecodeError as e:
            logger.error(f"Sales classifier JSON parse error: {e}")
            return self._safe_fallback("parse_error")
        except Exception as e:
            logger.error(f"Sales classifier error: {e}")
            return self._safe_fallback(str(e))

    def _safe_fallback(self, error: str) -> dict:
        return {
            "venta_cerrada": False,
            "datos_completos": False,
            "datos_faltantes": None,
            "datos_venta": {},
            "error": error,
        }
