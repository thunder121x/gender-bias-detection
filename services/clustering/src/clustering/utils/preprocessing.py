from __future__ import annotations

import re
from typing import Iterable

from pythainlp.corpus import thai_stopwords
from pythainlp.tokenize import word_tokenize

STOPWORDS = set(thai_stopwords())


def clean_text(text: str) -> str:
    """
    Strip urls, mentions, hashtags, and emoji while preserving Thai gender cues.
    Accepts anything string-like to avoid surprising failures upstream.
    """
    text = "" if text is None else str(text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    text = re.sub(r"#\S+", "", text)

    emoji_pattern = re.compile(
        "["  # keep ASCII to avoid encoding surprises
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols
        "\U0001F680-\U0001F6FF"  # transport
        "\U0001F1E0-\U0001F1FF"  # flags
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub("", text)
    return text.strip()


def thai_tokenize(text: str, stopwords: Iterable[str] | None = None) -> str:
    """
    Tokenize Thai text with newmm and drop common stopwords while keeping cues.
    """
    tokens = word_tokenize(text, engine="newmm")
    stopwords = STOPWORDS if stopwords is None else set(stopwords)
    tokens = [token for token in tokens if token and token not in stopwords]
    return " ".join(tokens)


def preprocess(text: str, stopwords: Iterable[str] | None = None) -> str:
    """
    Clean and tokenize Thai social content for downstream BERTopic modeling.
    """
    return thai_tokenize(clean_text(text), stopwords=stopwords)


__all__ = ["clean_text", "thai_tokenize", "preprocess", "STOPWORDS"]
