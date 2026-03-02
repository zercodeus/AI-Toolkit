"""
nlp.py — Natural Language Processing utilities.
"""

import math
import re
from collections import Counter


def tokenize(text: str, lowercase: bool = True) -> list:
    """
    Simple word tokenizer. Removes punctuation and splits on whitespace.

    Args:
        text:      Input string.
        lowercase: Convert tokens to lowercase (default True).

    Returns:
        List of word tokens.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")
    text = text.lower() if lowercase else text
    return re.findall(r'\b[a-zA-Z]+\b', text)


def remove_stopwords(tokens: list, stopwords: set = None) -> list:
    """Remove common English stopwords from a token list."""
    DEFAULT_STOPWORDS = {
        "a", "an", "the", "is", "it", "in", "on", "at", "to", "and",
        "or", "but", "of", "for", "with", "this", "that", "was", "be",
        "are", "have", "has", "had", "by", "from", "as", "not", "so",
    }
    stopwords = stopwords or DEFAULT_STOPWORDS
    return [t for t in tokens if t not in stopwords]


def term_frequency(tokens: list) -> dict:
    """Compute normalized term frequency for a list of tokens."""
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {word: count / total for word, count in counts.items()}


def inverse_document_frequency(corpus: list) -> dict:
    """
    Compute IDF for each term across a corpus.

    Args:
        corpus: List of token lists (one per document).

    Returns:
        Dict mapping term → IDF score.
    """
    n_docs = len(corpus)
    all_terms = set(term for doc in corpus for term in doc)
    idf = {}
    for term in all_terms:
        docs_with_term = sum(1 for doc in corpus if term in doc)
        idf[term] = math.log(n_docs / (1 + docs_with_term))
    return idf


def tfidf(tokens: list, corpus: list) -> dict:
    """
    Compute TF-IDF scores for tokens given a corpus.

    Args:
        tokens: Tokens of the current document.
        corpus: Full list of token lists.

    Returns:
        Dict mapping term → TF-IDF score.
    """
    tf = term_frequency(tokens)
    idf = inverse_document_frequency(corpus)
    return {term: tf.get(term, 0) * idf.get(term, 0) for term in tf}


def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """
    Compute cosine similarity between two numeric vectors.

    Returns a value in [0, 1] where 1 = identical direction.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must be the same length.")
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a ** 2 for a in vec_a))
    mag_b = math.sqrt(sum(b ** 2 for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)
