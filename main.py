from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

from agents.workflow import AgentWorkflow
from memory.postgres_memory import PostgresMemory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sales Agent API")
workflow = AgentWorkflow()
memory = PostgresMemory()


class ChatRequest(BaseModel):
    question: str
    session_id: str
    system_prompt: str = ""
    table_name: str = "kb_demo"
    memory_table: str = "demo_chat_db"


class ChatResponse(BaseModel):
    answer: str
    is_relevant: bool
    verification_report: Optional[str] = ""


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # 1. Cargar historial
        chat_history = memory.get_history(
            request.session_id,
            table_name=request.memory_table,
        )

        # 2. Ejecutar workflow
        result = await workflow.run(
            question=request.question,
            system_prompt=request.system_prompt,
            chat_history=chat_history,
            table_name=request.table_name,
        )

        # 3. Guardar en memoria
        memory.save_interaction(
            session_id=request.session_id,
            question=request.question,
            answer=result["answer"],
            table_name=request.memory_table,
        )

        return ChatResponse(
            answer=result["answer"],
            is_relevant=result["is_relevant"],
            verification_report=result.get("verification_report", ""),
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}