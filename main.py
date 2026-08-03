from config.gcp_auth import setup_google_credentials
setup_google_credentials()
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from agents.workflow import AgentWorkflow
from agents.sales_classifier import SalesClassifier
from memory.postgres_memory import PostgresMemory
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sales Agent API", docs_url=None, redoc_url=None, openapi_url=None)
workflow = AgentWorkflow()
classifier = SalesClassifier()
memory = PostgresMemory()

# API Key Security
api_key_header = APIKeyHeader(name="X-Api-Key")


async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != settings.AGENT_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key


class ChatRequest(BaseModel):
    question: str
    session_id: str
    system_prompt: str = ""
    table_name: str = "kb_demo"
    memory_table: str = "demo_chat_db"


class ChatResponse(BaseModel):
    answer: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, api_key: str = Depends(verify_api_key)):
    try:
        chat_history = memory.get_history(
            request.session_id,
            table_name=request.memory_table,
        )
        lima_tz = ZoneInfo("America/Lima")
        now_lima = datetime.now(lima_tz)
        dias = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
        meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", 
                "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
        dia_semana = dias[now_lima.weekday()]
        mes = meses[now_lima.month - 1]
        current_time_str = f"{dia_semana} {now_lima.day} de {mes} de {now_lima.year}, {now_lima.strftime('%H:%M')} horas (formato 24h)"
        result = await workflow.run(
            question=request.question,
            current_time=current_time_str,
            system_prompt=request.system_prompt,
            chat_history=chat_history,
            table_name=request.table_name,
        )

        memory.save_interaction(
            session_id=request.session_id,
            question=request.question,
            answer=result["answer"],
            table_name=request.memory_table,
        )

        return ChatResponse(answer=result["answer"])

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ClasificarVentaRequest(BaseModel):
    session_id: str
    sales_criteria: str
    memory_table: str = "demo_chat_db"


@app.post("/clasificar-venta")
async def clasificar_venta(request: ClasificarVentaRequest, api_key: str = Depends(verify_api_key)):
    try:
        chat_history = memory.get_history(
            request.session_id,
            table_name=request.memory_table,
        )
        result = await classifier.classify(
            chat_history=chat_history,
            sales_criteria=request.sales_criteria,
        )

        datos_venta = result.get("datos_venta") or {}
        if not datos_venta.get("numero_telefono"):
            datos_venta["numero_telefono"] = request.session_id
        result["datos_venta"] = datos_venta

        monto = datos_venta.get("monto") or "SIN_MONTO"
        fecha_voucher = datos_venta.get("fecha_voucher") or "SIN_FECHA"
        fecha_limpia = fecha_voucher.strip()
        if " - " in fecha_limpia:
            fecha_limpia = fecha_limpia.split(" - ")[0].strip()
        fecha_limpia = fecha_limpia.replace(" ", "").replace(".", "")
        monto_limpio = monto.replace("S/", "").replace(" ", "").strip()
        llave = f"{request.session_id}_{monto_limpio}_{fecha_limpia}"
        result["llave"] = llave

        return result

    except Exception as e:
        logger.error(f"Clasificar venta error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}
