"""Text utilities. Requires the `sklearn` extra."""

from __future__ import annotations

from sklearn.feature_extraction.text import CountVectorizer


def get_top_ngram(corpus: list[str], n: int | None = None, top_k: int = 10) -> list[tuple[str, int]]:
    """Return the `top_k` most frequent n-grams in `corpus`."""
    ngram_range = (n, n) if n else (1, 1)
    vectorizer = CountVectorizer(ngram_range=ngram_range).fit(corpus)
    bag_of_words = vectorizer.transform(corpus)
    sums = bag_of_words.sum(axis=0)
    counts = [(word, int(sums[0, idx])) for word, idx in vectorizer.vocabulary_.items()]
    counts.sort(key=lambda item: item[1], reverse=True)
    return counts[:top_k]
