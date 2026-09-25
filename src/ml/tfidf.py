from __future__ import annotations

from collections import Counter

import numpy as np

from src.ml.text_normalization import normalize_match_text


def _tokenize(text: str) -> list[str]:
    return normalize_match_text(text).split()


def _build_vocab(docs: list[str]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for doc in docs:
        for token in _tokenize(doc):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def _tf_vector(tokens: list[str], vocab: dict[str, int]) -> np.ndarray:
    vector = np.zeros(len(vocab), dtype=np.float32)
    counts = Counter(tokens)
    total = len(tokens) or 1
    for token, count in counts.items():
        idx = vocab.get(token)
        if idx is not None:
            vector[idx] = count / total
    return vector


def _idf_vector(docs: list[str], vocab: dict[str, int]) -> np.ndarray:
    n_docs = len(docs)
    df = np.zeros(len(vocab), dtype=np.float32)
    for doc in docs:
        seen: set[str] = set()
        for token in _tokenize(doc):
            if token in vocab and token not in seen:
                seen.add(token)
                df[vocab[token]] += 1.0
    return np.log((1.0 + n_docs) / (1.0 + df)) + 1.0


def tfidf_cosine_similarity(query_text: str, candidate_texts: list[str]) -> list[float]:
    """Быстрое TF-IDF косинусное сходство query с каждым кандидатом.

    Корпус строится из query + текстов кандидатов, поэтому IDF учитывает
    редкость токенов именно в текущем наборе кандидатов. Для коротких текстов
    (до ~50 кандидатов) вычисление занимает доли миллисекунды.
    """
    if not query_text or not candidate_texts:
        return [0.0] * len(candidate_texts)

    docs = [query_text, *candidate_texts]
    vocab = _build_vocab(docs)
    if not vocab:
        return [0.0] * len(candidate_texts)

    idf = _idf_vector(docs, vocab)
    query_tokens = _tokenize(query_text)
    query_vec = _tf_vector(query_tokens, vocab) * idf
    query_norm = float(np.linalg.norm(query_vec))
    if query_norm == 0.0:
        return [0.0] * len(candidate_texts)
    query_vec = query_vec / query_norm

    scores: list[float] = []
    for candidate_text in candidate_texts:
        cand_tokens = _tokenize(candidate_text)
        cand_vec = _tf_vector(cand_tokens, vocab) * idf
        cand_norm = float(np.linalg.norm(cand_vec))
        if cand_norm == 0.0:
            scores.append(0.0)
            continue
        scores.append(float(np.dot(query_vec, cand_vec / cand_norm)))
    return scores