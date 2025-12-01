from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd

from .pipeline import TopicModelingConfig, run_topic_modeling


def _parse_ngram_range(raw: str) -> Tuple[int, int]:
    parts = raw.replace("(", "").replace(")", "").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("ngram-range must be two integers separated by a comma, e.g. 1,2")
    try:
        start, end = (int(part.strip()) for part in parts)
    except ValueError:
        raise argparse.ArgumentTypeError("ngram-range values must be integers")
    return start, end


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run lightweight LDA topic modeling for Thai text (no BERTopic)."
    )
    parser.add_argument("--input", required=True, help="Path to CSV containing a text column.")
    parser.add_argument(
        "--text-column",
        default="text",
        help="Column name with raw text (default: text).",
    )
    parser.add_argument(
        "--output-csv",
        default="topicmodeling_output.csv",
        help="Where to write per-document topics (default: topicmodeling_output.csv).",
    )
    parser.add_argument("--n-topics", type=int, default=10, help="Number of topics to discover (default: 10).")
    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="Maximum vocabulary size for CountVectorizer (default: 5000).",
    )
    parser.add_argument("--min-df", type=int, default=3, help="Minimum document frequency for tokens (default: 3).")
    parser.add_argument(
        "--max-df",
        type=float,
        default=0.95,
        help="Ignore terms in more than this proportion of docs (default: 0.95).",
    )
    parser.add_argument(
        "--ngram-range",
        type=_parse_ngram_range,
        default=_parse_ngram_range("1,2"),
        help="Token ngram range, e.g. 1,2 for unigrams+bigrams (default: 1,2).",
    )
    parser.add_argument("--max-iter", type=int, default=20, help="Max iterations for LDA (default: 20).")
    parser.add_argument(
        "--learning-method",
        choices=["batch", "online"],
        default="batch",
        help="LDA learning method (default: batch).",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--top-words",
        type=int,
        default=10,
        help="How many top words to keep per topic for reporting (default: 10).",
    )
    args = parser.parse_args()

    config = TopicModelingConfig(
        n_topics=args.n_topics,
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_range=args.ngram_range,
        max_iter=args.max_iter,
        learning_method=args.learning_method,
        random_state=args.random_state,
        top_words=args.top_words,
    )

    input_path = Path(args.input)
    df = pd.read_csv(input_path)

    result = run_topic_modeling(df=df, text_column=args.text_column, config=config)
    result.df.to_csv(args.output_csv, index=False)

    print(f"Wrote topics to {args.output_csv}")
    for topic_id, words in result.topic_words.items():
        terms = ", ".join(word for word, _ in words[: args.top_words])
        print(f"Topic {topic_id}: {terms}")


if __name__ == "__main__":
    main()
