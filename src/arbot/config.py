"""All configuration comes from environment variables (or a local .env, never committed).
The LLM_* / EMBEDDING_* names are fixed by the assignment brief."""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_provider: str = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_timeout_s: float = 20.0
    llm_max_retries: int = 3
    llm_structured_method: str = "auto"     # auto | function_calling | json_schema
    use_system_certs: bool = True           # trust the OS certificate store (corporate proxies)
    # embeddings
    embedding_provider: str = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_api_key: str = ""
    # FAQ retrieval (cosine similarity; calibrate per embedding model)
    faq_direct_threshold: float = Field(0.82, description="at/above: return the stored answer verbatim")
    faq_min_threshold: float = Field(
        0.62, description="below: treat as not in the data (bge-small: right answers >= 0.675, unrelated <= 0.567)")
    chroma_dir: str = "chroma"
    # flow
    strict_flow: bool = False
    enable_lane_c: bool = True
    max_turns: int = 100
    lead_db_path: str = "leads.db"
    checkpoint_db: str = ""                 # SQLite path; empty = in-memory sessions
    turn_log_path: str = "turns.jsonl"      # one JSON line per turn; empty = off
    llm_price_in_per_mtok: float = 0.0      # USD per million tokens (0 = free tier)
    llm_price_out_per_mtok: float = 0.0
    # API
    api_rate_limit: int = 30                # requests per window per client
    api_rate_window_s: float = 60.0
    session_ttl_s: float = 7200.0
    max_sessions: int = 10000

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key) or self.llm_provider == "ollama"
