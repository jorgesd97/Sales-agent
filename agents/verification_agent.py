import google.generativeai as genai
import logging

from config.settings import settings
from google.generativeai.types import HarmCategory, HarmBlockThreshold

logger = logging.getLogger(__name__)


class VerificationAgent:
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(
            model_name=settings.GEMINI_FLASH_MODEL_LOW,
            generation_config={"temperature": 0, "max_output_tokens": 200},
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        )

    async def check(self, answer: str, context: str) -> dict:
        prompt = f"""
You are an AI assistant designed to verify the accuracy and relevance of answers based on provided context.

**Instructions:**
- Verify the following answer against the provided context.
- Check for:
  1. Direct/indirect factual support (YES/NO)
  2. Unsupported claims (list any if present)
  3. Contradictions (list any if present)
  4. Relevance to the question (YES/NO)

**Format:**
Supported: YES/NO
Unsupported Claims: [item1, item2, ...]
Contradictions: [item1, item2, ...]
Relevant: YES/NO

**Answer:** {answer}
**Context:**
{context}

**Respond ONLY with the above format.**
"""

        try:
            response = self.model.generate_content(prompt)
            report = response.text.strip()
            logger.info(f"Verification report generated")

            parsed = self._parse_report(report)
            return {
                "verification_report": report,
                "is_valid": parsed.get("supported") == "YES"
                and parsed.get("relevant") == "YES",
            }

        except Exception as e:
            logger.error(f"Verification error: {e}")
            return {
                "verification_report": "Verification failed",
                "is_valid": False,
            }

    def _parse_report(self, report: str) -> dict:
        parsed = {}
        for line in report.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip().upper()
                if key == "supported":
                    parsed["supported"] = value
                elif key == "relevant":
                    parsed["relevant"] = value
        return parsed