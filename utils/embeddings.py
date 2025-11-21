# utils/embeddings.py
import requests
import os
from dotenv import load_dotenv

load_dotenv()


def get_embeddings(text: str) -> list:
    """Получить эмбеддинги текста через Mistral API"""
    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError("MISTRAL_API_KEY not found in environment variables")

    url = "https://api.mistral.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistral-embed",
        "input": text
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        embeddings_data = response.json()
        return embeddings_data["data"][0]["embedding"]

    except Exception as e:
        print(f"Error getting embeddings: {e}")
        # Возвращаем нулевой вектор в случае ошибки
        return [0.0] * 1024


def analyze_semantic_similarity(query: str, paper_data: dict) -> dict:
    """Анализировать семантическую схожесть с помощью LLM"""
    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        return {"analysis": "Анализ недоступен", "key_points": []}

    prompt = f"""
    Проанализируй семантическую схожесть между запросом пользователя и научной статьей.

    Запрос пользователя: {query}

    Информация о статье:
    - Заголовок: {paper_data.get('title', 'Не указан')}
    - Год: {paper_data.get('year', 'Не указан')}
    - Библиография: {paper_data.get('bibtex', 'Не указана')}

    Проанализируй и предоставь:
    1. Основные точки соприкосновения
    2. Ключевые различия
    3. Практическую значимость связи

    Будь кратким и конкретным.
    """

    try:
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "mistral-small-latest",
            "messages": [
                {
                    "role": "system",
                    "content": "Ты эксперт по анализу научных текстов. Ты предоставляешь точный и лаконичный анализ схожести."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 400,
            "temperature": 0.4
        }

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        result = response.json()
        analysis = result["choices"][0]["message"]["content"].strip()

        return {
            "analysis": analysis,
            "key_points": extract_key_points(analysis)
        }

    except Exception as e:
        print(f"Error in semantic analysis: {e}")
        return {
            "analysis": "Семантический анализ недоступен",
            "key_points": ["Анализ временно недоступен"]
        }


def extract_key_points(analysis: str) -> list:
    """Извлечь ключевые пункты из анализа"""
    # Простая эвристика для извлечения ключевых пунктов
    lines = analysis.split('\n')
    key_points = []

    for line in lines:
        line = line.strip()
        if line.startswith(('-', '•', '—', '1.', '2.', '3.')):
            key_point = line.lstrip('-•—123.').strip()
            if key_point and len(key_point) > 10:
                key_points.append(key_point)

    return key_points[:5] if key_points else ["Ключевые пункты не выделены"]