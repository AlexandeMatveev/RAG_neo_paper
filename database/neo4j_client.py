# database/neo4j_client.py
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()


class Neo4jClient:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        self.driver = None
        self._connect()

    def _connect(self):
        """Установить соединение с Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            # Проверка соединения
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("Connected to Neo4j successfully")
        except Exception as e:
            print(f"Error connecting to Neo4j: {e}")
            self.driver = None

    def find_similar_papers(self, query_embedding: list, top_k: int = 10):
        """Найти схожие статьи по косинусной схожести"""
        if not self.driver:
            return []

        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (p:Paper)
                    WHERE p.embedding IS NOT NULL
                    WITH p, 
                         p.embedding AS emb1,
                         $query_embedding AS emb2
                    WITH p, 
                         reduce(s = 0.0, i IN range(0, size(emb1)-1) | 
                            s + emb1[i] * emb2[i]) AS dotProduct,
                         sqrt(reduce(s1 = 0.0, x IN emb1 | s1 + x * x)) AS norm1,
                         sqrt(reduce(s2 = 0.0, x IN emb2 | s2 + x * x)) AS norm2
                    WHERE norm1 > 0 AND norm2 > 0
                    WITH p, dotProduct / (norm1 * norm2) AS similarity
                    ORDER BY similarity DESC
                    LIMIT $top_k
                    RETURN p.title AS title,
                           p.bibtex AS bibtex,
                           p.year AS year,
                           p.link AS link,
                           p.paper_id AS paper_id,
                           similarity
                """, {
                    "query_embedding": query_embedding,
                    "top_k": top_k
                })

                return [dict(record) for record in result]
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []

    def get_connected_papers(self, paper_id: str):
        """Получить связанные статьи"""
        if not self.driver:
            return []

        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (p:Paper {paper_id: $paper_id})-[r]-(connected:Paper)
                    RETURN connected.title AS title,
                           connected.paper_id AS paper_id,
                           type(r) AS relationship_type
                    LIMIT 10
                """, {"paper_id": paper_id})

                return [dict(record) for record in result]
        except Exception as e:
            print(f"Error getting connections: {e}")
            return []

    def get_stats(self):
        """Получить статистику базы данных"""
        if not self.driver:
            return {"paper_count": 0}

        try:
            with self.driver.session() as session:
                result = session.run("MATCH (p:Paper) RETURN count(p) AS paper_count")
                return result.single().data()
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {"paper_count": 0}

    def close(self):
        """Закрыть соединение"""
        if self.driver:
            self.driver.close()