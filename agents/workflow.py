from langgraph.graph import StateGraph, END
from typing import TypedDict
import logging

from .relevance_checker import RelevanceChecker
from .research_agent import ResearchAgent
from .verification_agent import VerificationAgent
from retriever.supabase_retriever import SupabaseRetriever

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    question: str
    context: str
    draft_answer: str
    verification_report: str
    is_relevant: bool
    is_valid: bool
    system_prompt: str
    chat_history: str
    table_name: str
    retry_count: int


class AgentWorkflow:
    def __init__(self):
        self.relevance_checker = RelevanceChecker()
        self.researcher = ResearchAgent()
        self.verifier = VerificationAgent()
        self.retriever = SupabaseRetriever()
        self.compiled = self._build()

    def _build(self):
        workflow = StateGraph(AgentState)

        # Nodos
        workflow.add_node("retrieve", self._retrieve_step)
        workflow.add_node("check_relevance", self._check_relevance_step)
        workflow.add_node("research", self._research_step)
        workflow.add_node("verify", self._verify_step)

        # Flujo
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "check_relevance")
        workflow.add_conditional_edges(
            "check_relevance",
            self._decide_after_relevance,
            {"relevant": "research", "irrelevant": END},
        )
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
        return {"context": context}

    async def _check_relevance_step(self, state: AgentState) -> dict:
        classification = await self.relevance_checker.check(
            question=state["question"],
            context=state["context"],
        )

        if classification in ("CAN_ANSWER", "PARTIAL"):
            return {"is_relevant": True}

        return {
            "is_relevant": True,
            "context": "",  # Sin contexto, el agente responde con el prompt
        }

    def _decide_after_relevance(self, state: AgentState) -> str:
        return "relevant" if state["is_relevant"] else "irrelevant"

    async def _research_step(self, state: AgentState) -> dict:
        answer = await self.researcher.generate(
            question=state["question"],
            context=state["context"],
            system_prompt=state.get("system_prompt", ""),
            chat_history=state.get("chat_history", ""),
        )
        return {"draft_answer": answer}

    async def _verify_step(self, state: AgentState) -> dict:
        result = await self.verifier.check(
            answer=state["draft_answer"],
            context=state["context"],
        )
        return {
            "verification_report": result["verification_report"],
            "is_valid": result["is_valid"],
            "retry_count": state.get("retry_count", 0) + 1,
        }

    def _decide_after_verify(self, state: AgentState) -> str:
        if state["is_valid"]:
            return "valid"
        if state.get("retry_count", 0) >= 2:
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
            is_relevant=False,
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
            "is_relevant": final_state["is_relevant"],
        }