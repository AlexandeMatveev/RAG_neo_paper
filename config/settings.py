import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Neo4j
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

    # Mistral AI
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

    # App
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))


settings = Settings()