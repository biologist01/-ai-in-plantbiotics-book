"""
Configuration settings for the RAG chatbot backend
"""
import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # API Keys
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    cohere_api_key: str = os.getenv("COHERE_API_KEY", "")
    
    # Qdrant Configuration
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "plant_biotech_ai_book")
    
    # Database Configuration
    database_url: str = os.getenv("DATABASE_URL", "")
    neon_database_url: str = os.getenv("NEON_DATABASE_URL", "")
    
    # Security
    secret_key: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    
    # Server Configuration
    port: int = int(os.getenv("PORT", "8000"))
    host: str = os.getenv("HOST", "0.0.0.0")

    # LiveKit (Voice Agent)
    # LIVEKIT_URL should be your LiveKit Cloud URL (wss://...) or self-hosted URL.
    livekit_url: str = os.getenv("LIVEKIT_URL", "")
    livekit_api_key: str = os.getenv("LIVEKIT_API_KEY", "")
    livekit_api_secret: str = os.getenv("LIVEKIT_API_SECRET", "")
    # Must match @server.rtc_session(agent_name=...) in the agent code.
    livekit_agent_name: str = os.getenv("LIVEKIT_AGENT_NAME", "plant-biotech-assistant")

    # RAG behavior
    # When enabled, the API will refuse to answer if no relevant context is found in the vector DB.
    strict_rag_mode: bool = os.getenv("STRICT_RAG_MODE", "false").strip().lower() in {"1", "true", "yes"}
    
    # Embedding Configuration
    # NOTE: Keep dimension in sync with the Cohere model used in app.vector_db
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "embed-english-light-v3.0")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))
    
    # Chunking Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    class Config:
        env_file = ".env"

settings = Settings()
