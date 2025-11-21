# services/search_service.py
import numpy as np
from utils.embeddings import get_embeddings
from utils.summarizer import generate_summary
from database.neo4j_client import Neo4jClient
from models.similarity import calculate_cosine_similarity


class SearchService:
    def __init__(self):
        self.neo4j_client = Neo4jClient()

    def vector_search(self, query: str, top_k: int = 10):
        """Векторный поиск по косинусной схожести"""
        try:
            # Получаем эмбеддинг запроса
            query_embedding = get_embeddings(query)

            # Используем Python implementation для надежности
            results = self.neo4j_client.find_similar_papers(query_embedding, top_k)

            # Добавляем анализ схожести и краткое описание
            enhanced_results = []
            for result in results:
                enhanced_result = self.enhance_result_with_analysis(result, query)
                enhanced_results.append(enhanced_result)

            return enhanced_results
        except Exception as e:
            print(f"Error in vector search: {e}")
            return []

    def enhance_result_with_analysis(self, result: dict, query: str) -> dict:
        """Добавить анализ схожести и краткое описание к результату"""
        try:
            # Анализ уровня схожести
            similarity_score = result.get('similarity', 0)
            similarity_analysis = self.analyze_similarity_level(similarity_score)

            # Генерация краткого описания
            summary = generate_summary(
                title=result.get('title', ''),
                bibtex=result.get('bibtex', ''),
                year=result.get('year', ''),
                query=query
            )

            # Добавляем анализ в результат
            result['similarity_analysis'] = similarity_analysis
            result['summary'] = summary
            result['similarity_percentage'] = f"{similarity_score * 100:.1f}%"

            return result

        except Exception as e:
            print(f"Error enhancing result: {e}")
            result['similarity_analysis'] = "Анализ недоступен"
            result['summary'] = "Краткое описание недоступно"
            result['similarity_percentage'] = f"{result.get('similarity', 0) * 100:.1f}%"
            return result

    def analyze_similarity_level(self, similarity: float) -> str:
        """Анализировать уровень схожести"""
        if similarity >= 0.9:
            return "🎯 Очень высокая схожесть - практически полное соответствие теме"
        elif similarity >= 0.7:
            return "✅ Высокая схожесть - тема напрямую связана с запросом"
        elif similarity >= 0.5:
            return "⚠️ Умеренная схожесть - тема частично соответствует запросу"
        elif similarity >= 0.3:
            return "📚 Низкая схожесть - косвенная связь с темой"
        else:
            return "🔍 Минимальная схожесть - слабая связь с запросом"

    def hybrid_search(self, query: str, top_k: int = 10):
        """Гибридный поиск (векторный + ключевые слова)"""
        return self.vector_search(query, top_k)

    def get_paper_connections(self, paper_id: str):
        """Получить связанные статьи из графа"""
        try:
            connections = self.neo4j_client.get_connected_papers(paper_id)
            # Добавляем анализ для связанных статей
            enhanced_connections = []
            for connection in connections:
                enhanced_conn = connection.copy()
                enhanced_conn['connection_type'] = self.get_connection_type(
                    connection.get('relationship_type', '')
                )
                enhanced_connections.append(enhanced_conn)
            return enhanced_connections
        except Exception as e:
            print(f"Error getting connections: {e}")
            return []

    def get_connection_type(self, relationship: str) -> str:
        """Определить тип связи между статьями"""
        relationship_map = {
            'CITES': '📖 Цитирует',
            'CITED_BY': '↩️ Цитируется в',
            'RELATED': '🔗 Связана с',
            'SIMILAR': '📊 Похожая тема'
        }
        return relationship_map.get(relationship, '🔗 Связана')

    def get_database_stats(self):
        """Получить статистику базы данных"""
        try:
            return self.neo4j_client.get_stats()
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {"paper_count": 0}