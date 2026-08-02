from langgraph.graph import StateGraph, END
from typing import TypedDict
import logging

from .research_agent import ResearchAgent
from .verification_agent import VerificationAgent
from retriever.supabase_retriever import SupabaseRetriever

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    question: str
    context: str
    draft_answer: str
    verification_report: str
    is_valid: bool
    system_prompt: str
    chat_history: str
    table_name: str
    retry_count: int


class AgentWorkflow:
    def __init__(self):
        self.researcher = ResearchAgent()
        self.verifier = VerificationAgent()
        self.retriever = SupabaseRetriever()
        self.compiled = self._build()

    def _build(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("retrieve", self._retrieve_step)
        workflow.add_node("research", self._research_step)
        workflow.add_node("verify", self._verify_step)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "research")
        workflow.add_edge("research", "verify")
        workflow.add_conditional_edges(
            "verify",
            self._decide_after_verify,
            {"valid": END, "retry": "research", "fail": END},
        )

        return workflow.compile()

    async def _retrieve_step(self, state: AgentState) -> dict:
        documents = await self.retriever.search(
            query=state["question"],
            table_name=state.get("table_name", "kb_demo"),
        )
        context = self.retriever.format_context(documents)
        logger.info(f"Retrieved {len(documents)} chunks")
        logger.info(f"Context generated: {len(context)} chars")  # ← agregar esto
        # Ver qué chunks trajo
        for i, doc in enumerate(documents):
            logger.info(f"Chunk {i}: {doc['content'][:100]}...")
        return {"context": context}

    async def _research_step(self, state: AgentState) -> dict:
        logger.info(f"Context length: {len(state['context'])} chars")
        logger.info(f"Context preview: {state['context'][:200]}...")

        retry_count = state.get("retry_count", 0)
        feedback = ""
        if retry_count > 0:
            feedback = state.get("verification_report", "")
            logger.info(f"Retry with correction feedback: {feedback}")

        answer = await self.researcher.generate(
            question=state["question"],
            context=state["context"],
            system_prompt=state.get("system_prompt", ""),
            chat_history=state.get("chat_history", ""),
            correction_feedback=feedback,
        )
        logger.info(f"Draft answer: {answer[:150]}...")
        return {"draft_answer": answer}

    async def _verify_step(self, state: AgentState) -> dict:
        result = await self.verifier.check(
            answer=state["draft_answer"],
            question=state["question"],
            chat_history=state.get("chat_history", ""),
            system_prompt=state.get("system_prompt", ""),
        )
        return {
            "verification_report": result["verification_report"],
            "is_valid": result["is_valid"],
            "retry_count": state.get("retry_count", 0) + 1,
        }

    def _decide_after_verify(self, state: AgentState) -> str:
        if state["is_valid"]:
            return "valid"
        if state.get("retry_count", 0) >= 1:
            logger.warning("Max retries reached, ending workflow")
            return "fail"
        logger.info("Verification failed, retrying research")
        return "retry"

    async def run(
        self,
        question: str,
        system_prompt: str = "",
        chat_history: str = "",
        table_name: str = "kb_demo",
    ) -> dict:
        initial_state = AgentState(
            question=question,
            context="",
            draft_answer="",
            verification_report="",
            is_valid=False,
            system_prompt=system_prompt,
            chat_history=chat_history,
            table_name=table_name,
            retry_count=0,
        )

        final_state = await self.compiled.ainvoke(initial_state)

        return {
            "answer": final_state["draft_answer"],
            "verification_report": final_state.get("verification_report", ""),
        }