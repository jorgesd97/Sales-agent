from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Gemini
    GEMINI_API_KEY: str
    GEMINI_FLASH_MODEL_LOW: str = "gemini-2.5-flash"
    GEMINI_FLASH_MODEL_HIGH: str = "gemini-2.5-flash"

    # Supabase Edge Function
    SUPABASE_URL: str
    SUPABASE_ANON_KEY: str
    SUPABASE_EDGE_FUNCTION_URL: str = ""

    # Postgres Memory
    POSTGRES_CONNECTION_STRING: str

    # Defaults
    DEFAULT_TABLE_NAME: str = "kb_demo"
    DEFAULT_MATCH_COUNT: int = 3

    # API
    AGENT_API_KEY: str

    class Config:
        env_file = ".env"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.SUPABASE_EDGE_FUNCTION_URL:
            self.SUPABASE_EDGE_FUNCTION_URL = (
                f"{self.SUPABASE_URL}/functions/v1/hybrid-search"
            )


settings = Settings()