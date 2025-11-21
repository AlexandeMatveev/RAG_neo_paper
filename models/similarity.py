# models/similarity.py
import numpy as np


def calculate_cosine_similarity(vec1: list, vec2: list) -> float:
    """Вычислить косинусную схожесть между двумя векторами"""
    if not vec1 or not vec2:
        return 0.0

    v1 = np.array(vec1)
    v2 = np.array(vec2)

    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)