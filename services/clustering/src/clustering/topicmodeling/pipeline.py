from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from ..utils.preprocessing import preprocess


@dataclass
class TopicModelingConfig:
    """
    Configuration for lightweight topic modeling without BERTopic.
    """

    n_topics: int = 10
    max_features: int = 5000
    min_df: int = 3
    max_df: float = 0.95
    ngram_range: Tuple[int, int] = (1, 2)
    max_iter: int = 20
    learning_method: str = "batch"
    random_state: int = 42
    top_words: int = 10


@dataclass
class TopicModelingResult:
    model: LatentDirichletAllocation
    vectorizer: CountVectorizer
    df: pd.DataFrame
    topics: List[int]
    topic_distributions: List[List[float]]
    topic_words: Dict[int, List[Tuple[str, float]]] = field(default_factory=dict)

    def top_words_by_topic(self, n: int | None = None) -> Dict[int, List[str]]:
        """
        Convenience accessor for the top n words for each discovered topic.
        """
        limit = len(next(iter(self.topic_words.values()), [])) if n is None else n
        return {
            topic_id: [word for word, _ in words[:limit]]
            for topic_id, words in self.topic_words.items()
        }


def _build_vectorizer(config: TopicModelingConfig) -> CountVectorizer:
    return CountVectorizer(
        max_features=config.max_features,
        min_df=config.min_df,
        max_df=config.max_df,
        ngram_range=config.ngram_range,
    )


def _build_model(config: TopicModelingConfig) -> LatentDirichletAllocation:
    return LatentDirichletAllocation(
        n_components=config.n_topics,
        max_iter=config.max_iter,
        learning_method=config.learning_method,
        random_state=config.random_state,
    )


def _extract_topic_words(
    model: LatentDirichletAllocation,
    vectorizer: CountVectorizer,
    top_n: int,
) -> Dict[int, List[Tuple[str, float]]]:
    """
    Map topic id to its top terms (word, weight).
    """
    feature_names = vectorizer.get_feature_names_out()
    topic_words: Dict[int, List[Tuple[str, float]]] = {}
    for topic_idx, topic_weights in enumerate(model.components_):
        top_indices = topic_weights.argsort()[-top_n:][::-1]
        topic_words[topic_idx] = [
            (feature_names[i], float(topic_weights[i])) for i in top_indices
        ]
    return topic_words


def run_topic_modeling(
    df: pd.DataFrame,
    text_column: str = "text",
    config: TopicModelingConfig | None = None,
) -> TopicModelingResult:
    """
    Simple LDA-based topic modeling pipeline for Thai text without BERTopic.
    """
    if config is None:
        config = TopicModelingConfig()
    if text_column not in df.columns:
        raise ValueError(f"Missing text column '{text_column}' in dataframe")

    working_df = df.copy()
    working_df["clean"] = working_df[text_column].astype(str).apply(preprocess)
    texts: Sequence[str] = working_df["clean"].tolist()

    vectorizer = _build_vectorizer(config)
    dtm = vectorizer.fit_transform(texts)

    model = _build_model(config)
    topic_distributions = model.fit_transform(dtm)
    topics = topic_distributions.argmax(axis=1).tolist()
    working_df["topic"] = topics

    topic_words = _extract_topic_words(model, vectorizer, top_n=config.top_words)

    return TopicModelingResult(
        model=model,
        vectorizer=vectorizer,
        df=working_df,
        topics=topics,
        topic_distributions=topic_distributions.tolist(),
        topic_words=topic_words,
    )


__all__ = [
    "TopicModelingConfig",
    "TopicModelingResult",
    "run_topic_modeling",
]
