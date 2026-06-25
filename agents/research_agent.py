import google.generativeai as genai
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


class ResearchAgent:
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(
            model_name=settings.GEMINI_FLASH_MODEL_HIGH,
            generation_config={"temperature": 0.3, "max_output_tokens": 1300},
        )

    async def generate(
        self,
        question: str,
        context: str,
        system_prompt: str = "",
        chat_history: str = "",
    ) -> str:
        prompt = f"""
{system_prompt}

**Chat History:**
{chat_history}

**Instructions:**
- Answer the following question using ONLY the provided context.
- Be clear, concise, and factual.
- If the context doesn't contain enough information, say so honestly.
- Do NOT invent information that is not in the context.

**Question:** {question}

**Context:**
{context}

**Provide your answer below:**
"""

        try:
            response = self.model.generate_content(prompt)
            answer = response.text.strip()
            logger.info(f"Research agent generated answer for: '{question}'")
            return answer

        except Exception as e:
            logger.error(f"Research agent error: {e}")
            return "Lo siento, hubo un error al procesar tu consulta."