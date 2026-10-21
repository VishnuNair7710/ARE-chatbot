"""Provider-agnostic chat-model and embeddings construction from Settings.

Provider SDKs are imported lazily so a missing optional package only matters if that
provider is selected.
"""
from __future__ import annotations

from arbot.config import Settings

OPENAI_COMPATIBLE_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "mistral": "https://api.mistral.ai/v1",
}


def build_chat_model(s: Settings):
    """Return a LangChain chat model (temperature 0: extraction and grounded answers)."""
    p = s.llm_provider.lower()
    common = dict(model=s.llm_model, temperature=0)
    if p == "groq":
        from langchain_groq import ChatGroq
        extra = {"reasoning_effort": "low"} if "gpt-oss" in s.llm_model else {}  # reasoning models: keep it fast
        return ChatGroq(api_key=s.llm_api_key, timeout=s.llm_timeout_s, max_retries=0, **common, **extra)
    if p in ("openai", *OPENAI_COMPATIBLE_BASE_URLS):
        from langchain_openai import ChatOpenAI
        base = s.llm_base_url or OPENAI_COMPATIBLE_BASE_URLS.get(p) or None
        return ChatOpenAI(api_key=s.llm_api_key, base_url=base, timeout=s.llm_timeout_s,
                          max_retries=0, **common)
    if p == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(google_api_key=s.llm_api_key, timeout=s.llm_timeout_s,
                                      max_retries=0, **common)
    if p == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(base_url=s.llm_base_url or None, **common)
    raise ValueError(f"Unsupported LLM_PROVIDER: {s.llm_provider!r}")


def structured_method(s: Settings) -> str:
    """How to ask for structured output. Groq's gpt-oss models prefix tool names with 'functions.',
    which breaks LangChain's function-calling parser, so they use JSON-schema mode instead."""
    if s.llm_structured_method != "auto":
        return s.llm_structured_method
    return "json_schema" if (s.llm_provider.lower() == "groq" and "gpt-oss" in s.llm_model) else "function_calling"


def build_embedder(s: Settings):
    """Return an object with embed_documents(list[str]) and embed_query(str)."""
    p = s.embedding_provider.lower()
    if p == "local":
        return FastEmbedEmbedder(s.embedding_model)
    if p == "hash":  # offline lexical embeddings: no download, weaker on paraphrases
        from arbot.data.hash_embedder import HashEmbedder
        return HashEmbedder()
    if p == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=s.embedding_model or "text-embedding-3-small",
                                api_key=s.embedding_api_key or s.llm_api_key)
    if p == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=s.embedding_model or "models/gemini-embedding-001",
                                            google_api_key=s.embedding_api_key or s.llm_api_key)
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {s.embedding_provider!r}")


class FastEmbedEmbedder:
    """Free local embeddings (ONNX, no torch, no API key). Model weights are fetched once on
    first use and cached on disk."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._load().embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._load().query_embed(text))).tolist()
