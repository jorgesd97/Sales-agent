# sales_classifier.py

import asyncio
import logging
import re
from typing import List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator

from config.settings import settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# MODELOS PYDANTIC
#
# Con Structured Outputs el SDK garantiza que la respuesta es JSON válido y
# conforme al schema, así que estos validadores ya no son defensa contra
# formato roto: validan REGLAS DE NEGOCIO (fecha ISO real, pedido que empieza
# con cantidad, monto numérico limpio).
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
        endpoint = settings.AZURE_OPENAI_ENDPOINT.rstrip("/")
        self.client = OpenAI(
            base_url=f"{endpoint}/openai/v1/",
            api_key=settings.AZURE_OPENAI_API_KEY,
        )
        self.deployment = settings.AZURE_OPENAI_DEPLOYMENT

    async def classify(self, chat_history: str, sales_criteria: str) -> dict:
        instructions = self._build_instructions(sales_criteria)
        try:
            # El cliente de OpenAI es síncrono: se va a un hilo para no bloquear
            # el event loop de FastAPI.
            # NOTA: los modelos gpt-5.x rechazan `temperature` con HTTP 400.
            # Con Structured Outputs no hace falta.
            response = await asyncio.to_thread(
                lambda: self.client.responses.parse(
                    model=self.deployment,
                    instructions=instructions,
                    input=f"**Conversación completa:**\n{chat_history}",
                    text_format=SalesClassificationOutput,
                )
            )

            parsed = response.output_parsed
            if parsed is None:
                logger.warning("Sales classifier returned empty response")
                return self._safe_fallback("empty_response")

            output = parsed.model_dump()
            logger.info(
                f"Sales classification: venta_cerrada={output.get('venta_cerrada')}, "
                f"datos_completos={output.get('datos_completos')}"
            )
            return output

        except ValidationError as e:
            # El schema lo garantiza el SDK; esto es una regla de negocio rota
            # (fecha no ISO, pedido sin cantidad, etc.).
            logger.error(f"Sales classifier business-rule validation error: {e}")
            return self._safe_fallback(f"validation error: {str(e)}")
        except Exception as e:
            logger.error(f"Sales classifier error: {e}")
            return self._safe_fallback(str(e))

    def _build_instructions(self, sales_criteria: str) -> str:
        return f"""You are a sales classification agent. Analyze the conversation the user provides to determine if a sale was completed.

**Your tasks:**
1. Confirm whether there is a validated payment in the conversation (look for the marker [COMPROBANTE_PAGO]).
2. Check each criterion from the sales criteria against the conversation. Be STRICT: mark datos_completos as false if ANY criterion is missing.
3. Extract the sale data from the conversation. If a data point is NOT found in the conversation, set it as null and list it in datos_faltantes. Do NOT invent data.

Respect the description of every field of the output schema: it defines the exact
format expected for dates, times, amounts and the order line.

**Sales Criteria:**
{sales_criteria}
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
