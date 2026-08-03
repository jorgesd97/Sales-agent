from langgraph.graph import StateGraph, END
from typing import TypedDict
import logging

from .research_agent import ResearchAgent
from retriever.supabase_retriever import SupabaseRetriever

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    question: str
    current_time: str
    context: str
    draft_answer: str
    system_prompt: str
    chat_history: str
    table_name: str


class AgentWorkflow:
    def __init__(self):
        self.researcher = ResearchAgent()
        self.retriever = SupabaseRetriever()
        self.compiled = self._build()

    def _build(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("retrieve", self._retrieve_step)
        workflow.add_node("research", self._research_step)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "research")
        workflow.add_edge("research", END)
        return workflow.compile()

    async def _retrieve_step(self, state: AgentState) -> dict:
        documents = await self.retriever.search(
            query=state["question"],
            table_name=state.get("table_name", "kb_demo"),
        )
        context = self.retriever.format_context(documents)
        logger.info(f"Retrieved {len(documents)} chunks")
        logger.info(f"Context generated: {len(context)} chars")
        for i, doc in enumerate(documents):
            logger.info(f"Chunk {i}: {doc['content'][:100]}...")
        return {"context": context}

    async def _research_step(self, state: AgentState) -> dict:
        logger.info(f"Context length: {len(state['context'])} chars")
        answer = await self.researcher.generate(
            question=state["question"],
            current_time=state.get("current_time", ""),
            context=state["context"],
            system_prompt=state.get("system_prompt", ""),
            chat_history=state.get("chat_history", ""),
        )
        logger.info(f"Draft answer: {answer[:150]}...")
        return {"draft_answer": answer}

    async def run(
        self,
        question: str,
        current_time: str = "",
        system_prompt: str = "",
        chat_history: str = "",
        table_name: str = "kb_demo",
    ) -> dict:
        initial_state = AgentState(
            question=question,
            current_time=current_time,
            context="",
            draft_answer="",
            system_prompt=system_prompt,
            chat_history=chat_history,
            table_name=table_name,
        )
        final_state = await self.compiled.ainvoke(initial_state)
        return {"answer": final_state["draft_answer"]}
