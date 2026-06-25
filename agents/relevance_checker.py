import google.generativeai as genai
import logging

from config.settings import settings
from google.generativeai.types import HarmCategory, HarmBlockThreshold

logger = logging.getLogger(__name__)


class RelevanceChecker:
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(
            model_name=settings.GEMINI_FLASH_MODEL_LOW,
            generation_config={"temperature": 0, "max_output_tokens": 50},
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        )

    async def check(self, question: str, context: str) -> str:
        prompt = f"""
You are an AI relevance checker between a user's question and provided document content.

**Instructions:**
- Classify how well the document content addresses the user's question.
- Respond with only one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH.
- Do not include any additional text or explanation.
- If the question is a greeting or casual message (like "hola", "hi", "buenos dias"), respond with NO_MATCH.

**Labels:**
1) "CAN_ANSWER": The passages contain enough explicit information to fully answer the question.
2) "PARTIAL": The passages mention or discuss the question's topic but do not provide all the details needed for a complete answer.
3) "NO_MATCH": The passages do not discuss or mention the question's topic at all, OR the user is just greeting/chatting.

**Question:** {question}
**Passages:** {context}

**Respond ONLY with one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH**
"""

        try:
            response = self.model.generate_content(prompt)
            classification = response.text.strip().upper()
            logger.info(f"Relevance check: '{question}' -> {classification}")

            valid_labels = {"CAN_ANSWER", "PARTIAL", "NO_MATCH"}
            if classification not in valid_labels:
                return "NO_MATCH"

            return classification

        except Exception as e:
            logger.error(f"Relevance check error: {e}")
            return "NO_MATCH"