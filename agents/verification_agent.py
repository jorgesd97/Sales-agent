from google import genai
from google.genai import types
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


class VerificationAgent:
    def __init__(self):
        import os
        from google.oauth2 import service_account
        
        # 1. Cargamos explícitamente el archivo JSON que tu función setup_google_credentials() escribió en el disco
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        credentials = service_account.Credentials.from_service_account_file(
        creds_path, 
        scopes=scopes  # <-- ESTO SOLUCIONA EL "INVALID_SCOPE"
    )
        self.client = genai.Client(
        vertexai=True,
        project=settings.GOOGLE_CLOUD_PROJECT,
        location=settings.GOOGLE_CLOUD_LOCATION,
        credentials=credentials 
    )
        self.config = types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=300,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_NONE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_NONE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_NONE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_NONE",
                ),
            ]
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
            # 3. Llamamos al modelo desde el cliente, pasando el modelo y la configuración guardada
            response = self.client.models.generate_content(
                model=settings.GEMINI_FLASH_MODEL_LOW, # Asegúrate de que sea por ejemplo 'gemini-2.5-flash'
                contents=prompt,
                config=self.config
            )
            
            report = response.text.strip()
            logger.info("Verification report generated")

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