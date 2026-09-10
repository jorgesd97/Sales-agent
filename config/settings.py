from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- Azure OpenAI (reemplaza Vertex AI) ---
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_DEPLOYMENT: str = "gpt5mini-dodo"
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = "embeddings-dodo"

    # --- Azure AI Search (reemplaza la Edge Function `hybrid-search`) ---
    AZURE_SEARCH_ENDPOINT: str
    AZURE_SEARCH_ADMIN_KEY: str
    AZURE_SEARCH_INDEX_NAME: str = "dodo-knowledge-base"

    # --- Supabase: memoria + promociones (aún sin migrar) ---
    SUPABASE_URL: str
    SUPABASE_ANON_KEY: str
    POSTGRES_CONNECTION_STRING: str

    # --- Defaults ---
    DEFAULT_MATCH_COUNT: int = 3

    # --- API ---
    AGENT_API_KEY: str

    class Config:
        env_file = ".env"


settings = Settings()
