# sales_classifier.py  (o el nombre que tengas)

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
# 1. MODELOS PYDANTIC (ponlos aquí arriba, antes de la clase)
# ═══════════════════════════════════════════════════════════════

class DatosVenta(BaseModel):
    pedido: Optional[str] = Field(
        None,
        description="Formato: 'nro x descripcion'. Ej: '2 unidades de Perlita de 5 Litros para Plantas'. Múltiples items con \\n"
    )
    monto: Optional[str] = Field(None, description="Ej: S/ 70.80")
    fecha_entrega: Optional[str] = Field(
        None,
        description="Formato obligatorio: 'día. Mes año' en español. Ej: '11 ago. 2026'"
    )
    hora_entrega: Optional[str] = Field(
        None,
        description="Formato: 'HH:MM - HH:MM' si es rango, o 'HH:MM'. Ej: '19:00 - 23:00'"
    )
    nombre: Optional[str] = None
    numero_telefono: Optional[str] = Field(None, description="SIEMPRE null. El backend lo llena.")
    direccion: Optional[str] = None
    fecha_voucher: Optional[str] = Field(
        None,
        description="YYYY-MM-DD estricto. Ej: '2026-08-02'"
    )

    @field_validator('fecha_entrega')
    @classmethod
    def validate_fecha_entrega(cls, v):
        if v is None:
            return v
        if not re.match(r'^\d{1,2}\s+[a-z]{3,}\.\s*\d{4}$', v, re.IGNORECASE):
            raise ValueError("fecha_entrega debe ser 'día. Mes año'. Ej: '11 ago. 2026'")
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
            if not re.match(r'^\d+\s+x\s+.+', line):
                raise ValueError(f"Pedido mal formado: {line}")
        return '\n'.join(lines)


class SalesClassificationOutput(BaseModel):
    venta_cerrada: bool
    datos_completos: bool
    datos_faltantes: Optional[List[str]] = None
    datos_venta: DatosVenta
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════
# 2. TU CLASE SalesClassifier (abajo, usando los modelos)
# ═══════════════════════════════════════════════════════════════

class SalesClassifier:
    def __init__(self):
        # ... tu código actual de init ...
        pass

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
                    return self._safe_fallback("empty_response")

                raw = response.text.strip()
                raw = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE)
                raw = re.sub(r"\s*```$", "", raw)

                parsed = json.loads(raw)

                # AQUÍ USAS PYDANTIC PARA VALIDAR
                validated = SalesClassificationOutput(**parsed)

                result_dict = validated.model_dump()
                result_dict["datos_venta"]["numero_telefono"] = None  # Forzar null

                logger.info(f"OK: venta_cerrada={result_dict['venta_cerrada']}")
                return result_dict

            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"Intento {attempt + 1} falló: {e}")
                if attempt == 0:
                    prompt += f"\n\nERROR PREVIO: {str(e)}\nCorrige el JSON siguiendo el formato exacto."
                else:
                    return self._safe_fallback(f"validation_error: {str(e)}")
            except Exception as e:
                logger.error(f"Error inesperado: {e}")
                return self._safe_fallback(str(e))

    def _build_prompt(self, chat_history: str, sales_criteria: str) -> str:
        schema = SalesClassificationOutput.model_json_schema()
        return f"""You are a sales classification agent.

**OUTPUT FORMAT (STRICT JSON):**
{json.dumps(schema, indent=2)}

**CRITICAL RULES:**
- fecha_entrega: "día. Mes año"  →  Ej: "11 ago. 2026"
- hora_entrega: "HH:MM - HH:MM" o "HH:MM"  →  Ej: "19:00 - 23:00"
- pedido: cada línea "nro x descripcion"  →  Ej: "2 unidades de Perlita de 5 Litros para Plantas"
- fecha_voucher: YYYY-MM-DD  →  Ej: "2026-08-02"
- numero_telefono: null (siempre)
- Si falta un dato, usa null. NUNCA inventes datos.

**Sales Criteria:**
{sales_criteria}

**Conversation:**
{chat_history}

**Respond ONLY with raw JSON. No markdown, no ``` fences.**"""

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
                "numero_telefono": None,
                "direccion": None,
                "fecha_voucher": None,
            },
            "error": error,
        }