from google import genai
from google.genai import types
import json
import logging
import re

from config.settings import settings

logger = logging.getLogger(__name__)

VALID_STAGES = {
    "bienvenida", "objeciones", "logistica",
    "confirmacion_datos", "pago", "validacion", "agregar_producto",
}


class RouterAgent:
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
            max_output_tokens=100,
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

    async def classify_stage(self, chat_history: str, question: str, naturaleza_producto: str) -> dict:
        prompt = f"""You are a sales conversation router. Given the conversation history and the customer's latest message, determine which sales stage the conversation is currently in.

RULES:
- If the current message contains the marker [COMPROBANTE_PAGO] → "validacion"
- If payment info was already shown AND all required data is collected → "pago"
- If products are chosen AND delivery data is still missing → "confirmacion_datos"
- If the customer is interested but still has questions/objections → "objeciones"
- If the customer explicitly asks to ADD another product to the order (action, not just asking about it) → "agregar_producto"
- If the customer is new or just showing initial interest → "bienvenida"
- If the product is PHYSICAL and shipping cost hasn't been discussed yet, and the customer provided or is discussing their location → "logistica"

CRITICAL: Questions do NOT change the transactional stage. If a customer in "bienvenida" asks "do you accept Yape?", the stage stays "bienvenida" — the node will answer the payment question without advancing the sale. Only concrete ACTIONS advance the stage (giving data, confirming an order, sending a receipt).

Distinguish "asking about X" (informational, no stage change) from "doing X" (action, may change stage):
- "¿venden sustratos?" → informational, stay in current stage
- "agrégame un sustrato" → action, "agregar_producto"

Product type: {naturaleza_producto}
(If DIGITAL, never route to "logistica")

**Conversation history:**
{chat_history}

**Current customer message:**
{question}

Respond ONLY with valid JSON, no additional text, no markdown, no ```json fences:
{{"etapa": "..."}}

Valid values: bienvenida, objeciones, logistica, confirmacion_datos, pago, validacion, agregar_producto
"""

        try:
            response = self.client.models.generate_content(
                model=settings.GEMINI_FLASH_MODEL_LOW,
                contents=prompt,
                config=self.config,
            )

            if not response.text:
                logger.warning("Router returned empty response, defaulting to bienvenida")
                return {"etapa": "bienvenida"}

            raw = response.text.strip()
            raw = re.sub(r"^```json\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            result = json.loads(raw)
            etapa = result.get("etapa", "bienvenida")

            if etapa not in VALID_STAGES:
                logger.warning(f"Router returned invalid stage '{etapa}', defaulting to bienvenida")
                return {"etapa": "bienvenida"}

            if naturaleza_producto == "DIGITAL" and etapa == "logistica":
                logger.info("Product is DIGITAL, skipping logistica -> confirmacion_datos")
                return {"etapa": "confirmacion_datos"}

            logger.info(f"Router classified stage: {etapa}")
            return {"etapa": etapa}

        except json.JSONDecodeError as e:
            logger.error(f"Router JSON parse error: {e}")
            return {"etapa": "bienvenida"}
        except Exception as e:
            logger.error(f"Router error: {e}")
            return {"etapa": "bienvenida"}
