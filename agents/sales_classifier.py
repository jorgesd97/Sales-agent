# sales_classifier.py

from google import genai
from google.genai import types
import json
import logging
import re
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Optional, List
from config.settings import settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# MODELOS PYDANTIC
# ═══════════════════════════════════════════════════════════════

class DatosVenta(BaseModel):
    pedido: Optional[str] = Field(
        None,
        description="Formato obligatorio: 'nro x descripcion'. Ej: '2 unidades de Perlita de 5 Litros para Plantas'. "
                    "Si hay multiples items, separar con saltos de linea \\n. Ej: '2 x Perlita 5L\\n1 x Sustrato'"
    )
    monto: Optional[float] = Field(
        None,
        description="Monto TOTAL de la venta como numero puro, SIN simbolo de moneda, SIN comas. "
                    "Ej: 70.80 (no 'S/ 70.80', no '70,80')"
    )
    fecha_entrega: Optional[str] = Field(
        None,
        description="Formato ISO estricto YYYY-MM-DD. Solo fecha, sin hora, sin timezone. "
                    "Convierte cualquier fecha en español a este formato. Ej: '2026-08-11'"
    )
    hora_entrega: Optional[str] = Field(
        None,
        description="Formato: 'HH:MM - HH:MM' si es rango, o 'HH:MM' si es hora unica. Ej: '19:00 - 23:00', '14:30'"
    )
    nombre: Optional[str] = None
    direccion: Optional[str] = None
    fecha_voucher: Optional[str] = Field(
        None,
        description="YYYY-MM-DD estricto. Solo fecha, sin hora. Ej: '2026-08-11'"
    )

    @field_validator('monto', mode='before')
    @classmethod
    def validate_monto(cls, v):
        if v is None:
            return v
        # Defensa extra: si el LLM igual manda string con simbolo/comas, lo limpiamos.
        if isinstance(v, str):
            cleaned = re.sub(r'[^\d.]', '', v.replace(',', '.'))
            if not cleaned:
                raise ValueError(f"monto no contiene digitos validos: '{v}'")
            try:
                return float(cleaned)
            except ValueError:
                raise ValueError(f"monto no se pudo convertir a numero: '{v}'")
        return v

    @field_validator('fecha_entrega', 'fecha_voucher')
    @classmethod
    def validate_fecha_iso(cls, v):
        if v is None:
            return v
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', v):
            raise ValueError(
                f"Fecha debe ser formato ISO 'YYYY-MM-DD'. Recibido: '{v}'"
            )
        return v

    @field_validator('hora_entrega')
    @classmethod
    def validate_hora_entrega(cls, v):
        if v is None:
            return v
        if not re.match(r'^\d{2}:\d{2}(\s*-\s*\d{2}:\d{2})?$', v):
            raise ValueError("hora_entrega debe ser 'HH:MM - HH:MM' o 'HH:MM'")
        return v

    @field_validator('pedido')
    @classmethod
    def validate_pedido(cls, v):
        if v is None:
            return v
        lines = [line.strip() for line in str(v).split('\n') if line.strip()]
        for line in lines:
            if not re.match(r'^\d+\s+.+', line):
                raise ValueError(f"Cada linea de pedido debe empezar con cantidad. Fallo: {line}")
        return '\n'.join(lines)


class SalesClassificationOutput(BaseModel):
    venta_cerrada: bool
    datos_completos: bool
    datos_faltantes: Optional[List[str]] = None
    datos_venta: DatosVenta
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════
# CLASIFICADOR
# ═══════════════════════════════════════════════════════════════

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
            max_output_tokens=800,
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
        prompt = self._build_prompt(chat_history, sales_criteria)

        for attempt in range(2):
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
                raw = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE)
                raw = re.sub(r"\s*```$", "", raw)

                result = json.loads(raw)
                validated = SalesClassificationOutput(**result)
                output = validated.model_dump()

                logger.info(
                    f"Sales classification: venta_cerrada={output.get('venta_cerrada')}, "
                    f"datos_completos={output.get('datos_completos')}"
                )
                return output

            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"Sales classifier validation error (intento {attempt + 1}): {e}")
                if attempt == 0:
                    prompt += f"\n\n[ERROR DE FORMATO PREVIO]: {str(e)}\n"
                    prompt += "Corrige el JSON siguiendo EXACTAMENTE el formato y el ejemplo mostrados arriba. "
                    prompt += "Responde SOLO con el JSON corregido, sin texto adicional."
                else:
                    return self._safe_fallback(f"parse/validation error: {str(e)}")
            except Exception as e:
                logger.error(f"Sales classifier error: {e}")
                return self._safe_fallback(str(e))

    def _build_prompt(self, chat_history: str, sales_criteria: str) -> str:
        return f"""You are a sales classification agent. Analyze the following conversation to determine if a sale was completed.

**Your tasks:**
1. Confirm whether there is a validated payment in the conversation (look for the marker [COMPROBANTE_PAGO]).
2. Check each criterion from the sales criteria against the conversation. Be STRICT: mark datos_completos as false if ANY criterion is missing.
3. Extract the sale data from the conversation. If a data point is NOT found in the conversation, set it as null and list it in datos_faltantes. Do NOT invent data.

**CRITICAL OUTPUT FORMAT RULES (follow these EXACTLY):**
- `fecha_entrega`: MUST be ISO format "YYYY-MM-DD", NO time, NO timezone. Convert any Spanish date mentioned in the conversation to this format. Examples: "11 ago. 2026" -> "2026-08-11", "mañana" (resolve using current date reference) -> "2026-08-12"
- `hora_entrega`: MUST be "HH:MM - HH:MM" for time ranges, or "HH:MM" for single time. Examples: "19:00 - 23:00", "14:30"
- `pedido`: MUST start with the quantity. Examples: "2 x Perlita de 5 Litros para Plantas". If multiple items, use \\n between lines.
- `fecha_voucher`: extract ONLY the date (no time) and return in strict format YYYY-MM-DD. Convert Spanish month names/abbreviations to numbers (e.g. "02 Ago. 2026" -> "2026-08-02", "15 de julio de 2026" -> "2026-07-15"). If the date cannot be determined, use null.
- `monto`: MUST be a plain NUMBER (JSON number type, not string), WITHOUT currency symbol, WITHOUT thousand separators. Examples: 70.80 (NOT "S/ 70.80", NOT "70,80", NOT "S/70.80")
- If any data is missing, use null. NEVER invent data.

**EXAMPLE OUTPUT:**
{{
  "venta_cerrada": true,
  "datos_completos": true,
  "datos_faltantes": null,
  "datos_venta": {{
    "pedido": "2 x Perlita de 5 Litros para Plantas",
    "monto": 70.80,
    "fecha_entrega": "2026-08-11",
    "hora_entrega": "19:00 - 23:00",
    "nombre": "Eduardo Soto",
    "direccion": "miraflores av mariscal caceres 123 lote 8",
    "fecha_voucher": "2026-08-11"
  }}
}}

**Sales Criteria:**
{sales_criteria}

**Complete Conversation:**
{chat_history}

**Respond ONLY with valid JSON, no additional text, no markdown, no ```json fences. Only the JSON object:**
"""

    def _safe_fallback(self, error: str) -> dict:
        return {
            "venta_cerrada": False,
            "datos_completos": False,
            "datos_faltantes": None,
            "datos_venta": {
                "pedido": None,
                "monto": None,
                "fecha_entrega": None,
                "hora_entrega": None,
                "nombre": None,
                "direccion": None,
                "fecha_voucher": None,
            },
            "error": error,
        }