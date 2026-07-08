import psycopg
import json
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


class PostgresMemory:
    def __init__(self):
        self.conn_string = settings.POSTGRES_CONNECTION_STRING

    def _get_connection(self):
        return psycopg.connect(self.conn_string)

    def get_history(self, session_id: str, table_name: str = "demo_chat_db", limit: int = 10) -> str:
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT msg_type, message, msg_ts
                FROM {table_name} 
                WHERE session_id = %s 
                ORDER BY id DESC 
                LIMIT %s
                """,
                (session_id, limit),
            )
            rows = cur.fetchall()
            cur.close()
            conn.close()

            rows.reverse()

            history = ""
            for row in rows:
                msg_type = row[0]
                message = row[1]
                msg_ts = row[2]
                # message es jsonb, puede venir como dict o string
                if isinstance(message, str):
                    try:
                        message = json.loads(message)
                    except json.JSONDecodeError:
                        pass

                if isinstance(message, dict):
                    text = message.get("content", "")
                    
                else:
                    text = str(message)

                if not text:
                    continue
                timestamp_str = msg_ts.strftime("%Y-%m-%d %H:%M") if msg_ts else "Fecha desconocida"
                if msg_type == "human":
                    history += f"[{timestamp_str}] Human: {text}\n"
                elif msg_type == "ai":
                    history += f"[{timestamp_str}] AI: {text}\n"

            logger.info(f"Loaded {len(rows)} messages for session {session_id}")
            return history.strip()

        except Exception as e:
            logger.error(f"Memory load error: {e}")
            return ""

    def save_message(self, session_id: str, msg_type: str, content: str, table_name: str = "demo_chat_db"):
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            message = json.dumps({
                "type": msg_type,
                "content": content,
            })
            cur.execute(
                f"""
                INSERT INTO {table_name} (session_id, message)
                VALUES (%s, %s)
                """,
                (session_id, message),
            )
            conn.commit()
            cur.close()
            conn.close()
            logger.info(f"Saved {msg_type} message for session {session_id}")

        except Exception as e:
            logger.error(f"Memory save error: {e}")

    def save_interaction(self, session_id: str, question: str, answer: str, table_name: str = "demo_chat_db"):
        self.save_message(session_id, "human", question, table_name)
        self.save_message(session_id, "ai", answer, table_name)
