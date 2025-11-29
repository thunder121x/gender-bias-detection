from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .pipeline import DEFAULT_EMBEDDING_MODEL, TopicPipelineConfig, run_pipeline


def _parse_nr_topics(raw: str) -> int | str | None:
    if raw.lower() == "none":
        return None
    if raw.lower() == "auto":
        return "auto"
    return int(raw)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Thai BERTopic clustering for gender-bias mining."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to CSV containing a text column.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Column name with raw text (default: text).",
    )
    parser.add_argument(
        "--output-csv",
        default="bertopic_output.csv",
        help="Where to write per-document topics (default: bertopic_output.csv).",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Optional directory to save the fitted BERTopic model.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"SentenceTransformer model name (default: {DEFAULT_EMBEDDING_MODEL}).",
    )
    parser.add_argument(
        "--min-topic-size",
        type=int,
        default=30,
        help="Minimum cluster size passed to BERTopic (default: 30).",
    )
    parser.add_argument(
        "--nr-topics",
        default="auto",
        help="Target number of topics: auto, none, or integer (default: auto).",
    )
    parser.add_argument(
        "--reduce-to",
        type=int,
        default=25,
        help="Optional post hoc topic reduction target (set 0 to skip).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="How many examples to sample per bias topic (default: 200).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42).",
    )
    args = parser.parse_args()

    config = TopicPipelineConfig(
        embedding_model=args.embedding_model,
        min_topic_size=args.min_topic_size,
        nr_topics=_parse_nr_topics(args.nr_topics),
        reduce_to=None if args.reduce_to <= 0 else args.reduce_to,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    input_path = Path(args.input)
    df = pd.read_csv(input_path)

    result = run_pipeline(
        df=df,
        text_column=args.text_column,
        config=config,
    )
    result.save(args.output_csv, model_dir=args.model_dir)

    print(f"Wrote topics to {args.output_csv}")
    if args.model_dir:
        print(f"Saved BERTopic model to {args.model_dir}")
    print(f"Detected bias-related topics: {result.bias_topics}")


if __name__ == "__main__":
    main()
