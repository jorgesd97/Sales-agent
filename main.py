from config.gcp_auth import setup_google_credentials
setup_google_credentials()
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Optional
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from agents.workflow import AgentWorkflow
from memory.postgres_memory import PostgresMemory
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sales Agent API", docs_url=None, redoc_url=None, openapi_url=None)
workflow = AgentWorkflow()
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
    verification_report: Optional[str] = ""


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
        question_with_time = (
            f"[Fecha y hora actual en Lima, Perú: {current_time_str}]\n\n"
            f"{request.question}"
        )
        result = await workflow.run(
            question=question_with_time,
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

        return ChatResponse(
            answer=result["answer"],
            verification_report=result.get("verification_report", ""),
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}
