"""Example: run the BERTopic clustering pipeline on scraped text.

The input CSV (postprocessed_output.csv, ~80MB) is not stored in git;
it is produced by the scraper post-processing step — pass its path as
the first argument.
"""

import sys

import pandas as pd
from clustering import run_pipeline, TopicPipelineConfig

if len(sys.argv) < 2:
    sys.exit("usage: python run_pipeline.py <path/to/postprocessed_output.csv>")

df = pd.read_csv(sys.argv[1])
config = TopicPipelineConfig(
    embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    min_topic_size=30,
    reduce_to=25,
)

result = run_pipeline(df, text_column="text", config=config)
result.save("bertopic_output.csv", model_dir="bertopic_thai_gender_bias")

print("Bias topics:", result.bias_topics)
print("Sampled sentences from the first bias topic:", result.biased_samples.get(result.bias_topics[0], []))
