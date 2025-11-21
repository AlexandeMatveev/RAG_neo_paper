# app.py
import streamlit as st
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from services.search_service import SearchService
from utils.embeddings import get_embeddings, analyze_semantic_similarity
import os
from dotenv import load_dotenv

load_dotenv()


class StreamlitApp:
    def __init__(self):
        self.search_service = SearchService()

    def setup_page(self):
        st.set_page_config(
            page_title="Умный поиск научных статей",
            page_icon="🔍",
            layout="wide"
        )
        st.title("🔍 Умный поиск научных статей")
        st.markdown("Поиск схожих статей с AI-анализом и краткими описаниями")

    def search_interface(self):
        col1, col2 = st.columns([2, 1])

        with col1:
            query = st.text_area(
                "Введите запрос для поиска:",
                placeholder="Опишите тему исследования, концепцию или проблему...",
                height=100
            )

        with col2:
            search_type = st.selectbox(
                "Тип поиска:",
                ["Векторный поиск", "Гибридный поиск"]
            )
            top_k = st.slider("Количество результатов:", 1, 20, 10)
            show_analysis = st.checkbox("Показать детальный анализ", value=True)

        if st.button("🔍 Найти статьи", type="primary"):
            if query:
                with st.spinner("Ищем статьи и анализируем схожесть..."):
                    if search_type == "Векторный поиск":
                        results = self.search_service.vector_search(query, top_k)
                    else:
                        results = self.search_service.hybrid_search(query, top_k)

                    self.display_results(results, query, show_analysis)
            else:
                st.warning("Пожалуйста, введите запрос для поиска")

    def display_results(self, results, query, show_analysis):
        if not results:
            st.info("По вашему запросу ничего не найдено")
            return

        st.subheader(f"📊 Результаты поиска для: '{query}'")
        st.write(f"Найдено статей: {len(results)}")

        for i, result in enumerate(results, 1):
            with st.expander(
                    f"📄 {result.get('title', 'Без названия')} | Схожесть: {result.get('similarity_percentage', '0%')}",
                    expanded=i == 1):
                self.display_paper_details(result, show_analysis, i)

    def display_paper_details(self, result, show_analysis, index):
        # Основная информация
        col1, col2 = st.columns([3, 1])

        with col1:
            if 'title' in result:
                st.markdown(f"### {result['title']}")

            # Метрики схожести
            similarity_score = result.get('similarity', 0)
            similarity_analysis = result.get('similarity_analysis', '')

            st.markdown(f"**🎯 Уровень схожести:** {similarity_analysis}")
            st.markdown(f"**📈 Количественная оценка:** {result.get('similarity_percentage', '0%')}")

            # Прогресс-бар схожести
            st.progress(float(similarity_score))

        with col2:
            if 'year' in result:
                st.metric("Год публикации", result['year'])

            if 'link' in result and result['link']:
                st.markdown(f"[📎 Полный текст]({result['link']})")

            if st.button("🔍 Граф связей", key=f"graph_{result.get('paper_id', f'unknown_{index}')}"):
                self.show_graph_connections(result.get('paper_id'))

        # Краткое описание
        if 'summary' in result:
            st.markdown("---")
            st.markdown("#### 📝 Краткое описание")
            st.info(result['summary'])

        # Детальный анализ (если включен)
        if show_analysis and result.get('similarity', 0) > 0.3:
            st.markdown("---")
            st.markdown("#### 🔍 Детальный анализ схожести")

            col_analysis1, col_analysis2 = st.columns(2)

            with col_analysis1:
                st.markdown("**📊 Метрики:**")
                st.write(f"- Косинусная схожесть: `{result.get('similarity', 0):.4f}`")
                st.write(f"- Нормализованная оценка: `{result.get('similarity_percentage', '0%')}`")

                # Дополнительная информация о статье
                if 'bibtex' in result:
                    st.markdown("**📚 Источник:**")
                    st.write(result['bibtex'])

            with col_analysis2:
                st.markdown("**🎯 Рекомендация:**")
                if result.get('similarity', 0) >= 0.7:
                    st.success("✅ Высоко релевантная статья - рекомендуется к изучению")
                elif result.get('similarity', 0) >= 0.5:
                    st.warning("⚠️ Умеренная релевантность - может быть полезной")
                else:
                    st.info("📚 Косвенная связь - для общего ознакомления")

        st.markdown("---")

    def show_graph_connections(self, paper_id):
        if paper_id:
            with st.spinner("Загружаем связи..."):
                connections = self.search_service.get_paper_connections(paper_id)

            if connections:
                st.subheader("🔗 Связанные статьи")

                for connection in connections:
                    col_conn1, col_conn2 = st.columns([3, 1])

                    with col_conn1:
                        st.write(f"**{connection.get('title', 'Без названия')}**")

                    with col_conn2:
                        st.write(f"*{connection.get('connection_type', 'Связь')}*")
            else:
                st.info("Связи с другими статьями не найдены")

    def run(self):
        self.setup_page()
        self.search_interface()

        # Боковая панель с информацией
        with st.sidebar:
            st.header("ℹ️ О системе")
            st.markdown("""
            **AI-функции системы:**
            - 🔍 Векторный поиск с косинусной схожестью
            - 📝 AI-генерация кратких описаний
            - 🎯 Анализ уровня релевантности
            - 🔗 Графовый поиск связей
            - 📊 Семантический анализ схожести
            """)

            st.header("📊 Статистика")
            try:
                stats = self.search_service.get_database_stats()
                st.metric("Статей в базе", stats.get('paper_count', 0))

                # Информация о модели
                st.header("🤖 Модель")
                st.markdown("""
                - **Mistral AI** для эмбеддингов
                - **Mistral Small** для описаний
                - **Neo4j** для графового поиска
                """)

            except Exception as e:
                st.error("Статистика недоступна")


if __name__ == "__main__":
    app = StreamlitApp()
    app.run()