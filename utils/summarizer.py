# utils/summarizer.py
import requests
import os
from dotenv import load_dotenv

load_dotenv()


def generate_summary(title: str, bibtex: str, year: str, query: str) -> str:
    """Сгенерировать краткое описание статьи с помощью Mistral AI"""
    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        return "❌ API ключ не настроен"

    # Формируем промпт для генерации описания
    prompt = f"""
    Сгенерируй краткое описание научной статьи на основе следующей информации:

    Заголовок: {title}
    Год: {year}
    Библиографическая ссылка: {bibtex}

    Запрос пользователя: {query}

    Создай краткое описание (2-3 предложения), которое:
    1. Объясняет, о чем эта статья
    2. Показывает, как она связана с запросом пользователя
    3. Выделяет ключевые аспекты

    Ответ должен быть на русском языке, информативным и лаконичным.
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
                    "content": "Ты помощник для анализа научных статей. Ты создаешь краткие и информативные описания."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 300,
            "temperature": 0.3
        }

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        result = response.json()
        summary = result["choices"][0]["message"]["content"].strip()

        return summary

    except Exception as e:
        print(f"Error generating summary: {e}")
        return f"📄 Статья: {title} ({year}). Связь с запросом требует дополнительного анализа."