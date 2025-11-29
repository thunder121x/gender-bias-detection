from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from .preprocessing import preprocess

DEFAULT_BIAS_KEYWORDS: Tuple[str, ...] = (
    "ผู้หญิง",
    "ผู้ชาย",
    "เมีย",
    "ผัว",
    "ตุ๊ด",
    "เกย์",
    "กะเทย",
    "สาว",
    "อี",
    "มัน",
    "อารมณ์",
    "งอแง",
    "อ่อนแอ",
    "แรง",
    "หึง",
    "เจ้าชู้",
    "ทำงาน",
    "ทำอาหาร",
    "หาเงิน",
)
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class TopicPipelineConfig:
    """
    Tunable knobs for the Thai BERTopic pipeline.
    """

    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    min_df: int = 3
    ngram_range: Tuple[int, int] = (1, 2)
    min_topic_size: int = 30
    nr_topics: int | str | None = "auto"
    reduce_to: int | None = 25
    sample_size: int = 200
    seed: int = 42


@dataclass
class PipelineResult:
    topic_model: BERTopic
    topics: List[int]
    probabilities: List[List[float]] | None
    df: pd.DataFrame
    clean_df: pd.DataFrame
    bias_topics: List[int] = field(default_factory=list)
    biased_samples: Dict[int, List[str]] = field(default_factory=dict)

    def save(self, output_csv: str, model_dir: str | None = None) -> None:
        """
        Persist clustering results and, optionally, the fitted topic model.
        """
        self.df.to_csv(output_csv, index=False)
        if model_dir:
            self.topic_model.save(model_dir)


def build_topic_model(config: TopicPipelineConfig) -> BERTopic:
    """
    Construct a BERTopic model configured for Thai gender-bias mining.
    """
    embedding_model = SentenceTransformer(config.embedding_model)
    vectorizer = CountVectorizer(
        ngram_range=config.ngram_range,
        min_df=config.min_df,
    )
    return BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer,
        language="thai",
        calculate_probabilities=True,
        min_topic_size=config.min_topic_size,
        nr_topics=config.nr_topics,
    )


def detect_bias_topics(
    topic_model: BERTopic, bias_keywords: Iterable[str]
) -> List[int]:
    """
    Identify topics containing any gender/sexism cue words.
    """
    bias_keywords = list(bias_keywords)
    matched_topics: List[int] = []
    for topic_id, words in topic_model.get_topics().items():
        if topic_id == -1:
            continue
        word_string = " ".join([word for word, _ in words])
        if any(keyword in word_string for keyword in bias_keywords):
            matched_topics.append(topic_id)
    return matched_topics


def extract_samples(
    df: pd.DataFrame, topic_id: int, n: int, seed: int
) -> List[str]:
    """
    Sample example documents from a specific topic for annotation.
    """
    topic_df = df[df["topic"] == topic_id]
    if topic_df.empty:
        return []
    sampled = topic_df.sample(min(n, len(topic_df)), random_state=seed)
    return sampled["text"].astype(str).tolist()


def run_pipeline(
    df: pd.DataFrame,
    text_column: str = "text",
    bias_keywords: Sequence[str] | None = None,
    config: TopicPipelineConfig | None = None,
) -> PipelineResult:
    """
    Full BERTopic pipeline: preprocess -> model -> bias topic extraction.
    """
    if config is None:
        config = TopicPipelineConfig()
    if bias_keywords is None:
        bias_keywords = DEFAULT_BIAS_KEYWORDS

    if text_column not in df.columns:
        raise ValueError(f"Missing text column '{text_column}' in dataframe")

    working_df = df.copy()
    working_df["clean"] = working_df[text_column].astype(str).apply(preprocess)
    texts = working_df["clean"].tolist()

    topic_model = build_topic_model(config)
    topics, probabilities = topic_model.fit_transform(texts)

    if config.reduce_to:
        topic_model = topic_model.reduce_topics(
            texts,
            topics=topics,
            probabilities=probabilities,
            nr_topics=config.reduce_to,
        )
        probabilities = topic_model.probabilities_

    doc_info = topic_model.get_document_info(texts)
    working_df["topic"] = doc_info["Topic"].tolist()
    clean_df = working_df[working_df["topic"] != -1].copy()

    bias_topics = detect_bias_topics(topic_model, bias_keywords)
    biased_samples = {
        topic_id: extract_samples(
            clean_df, topic_id, n=config.sample_size, seed=config.seed
        )
        for topic_id in bias_topics
    }

    return PipelineResult(
        topic_model=topic_model,
        topics=working_df["topic"].tolist(),
        probabilities=probabilities,
        df=working_df,
        clean_df=clean_df,
        bias_topics=bias_topics,
        biased_samples=biased_samples,
    )


__all__ = [
    "TopicPipelineConfig",
    "PipelineResult",
    "run_pipeline",
    "build_topic_model",
    "detect_bias_topics",
    "extract_samples",
    "DEFAULT_BIAS_KEYWORDS",
    "DEFAULT_EMBEDDING_MODEL",
]
