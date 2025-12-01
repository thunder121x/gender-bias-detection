# Clustering Service

Thai topic modeling toolkit for gender-bias mining. Includes BERTopic pipeline plus a lightweight LDA-based alternative, shared preprocessing, CLIs for both, and helpers for programmatic use.

## Quickstart

```bash
cd services/clustering
pip install -e .
# BERTopic pipeline (Thai embeddings + bias detection)
python -m clustering.bertopic --input ../testset.csv --text-column text --output-csv assets/bertopic_output.csv --model-dir assets/bertopic_thai_gender_bias

# Lightweight LDA topic modeling (no BERTopic dependency)
python -m clustering.topicmodeling --input ../testset.csv --text-column text --output-csv assets/topicmodeling_output.csv --n-topics 12
```

This writes per-document topics to `assets/bertopic_output.csv`, saves the trained model, and prints detected bias topics.

## Programmatic Use

```python
import pandas as pd
from clustering import (
    TopicModelingConfig,
    TopicPipelineConfig,
    run_pipeline,
    run_topic_modeling,
)

df = pd.read_csv("path/to/data.csv")
config = TopicPipelineConfig(
    embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    min_topic_size=30,
    reduce_to=25,
)

result = run_pipeline(df, text_column="text", config=config)
result.save("bertopic_output.csv", model_dir="bertopic_thai_gender_bias")

print("Bias topics:", result.bias_topics)
print("Sampled sentences from the first bias topic:", result.biased_samples.get(result.bias_topics[0], []))

# LDA-based topic modeling (no BERTopic)
lda_config = TopicModelingConfig(n_topics=12, ngram_range=(1, 2))
lda_result = run_topic_modeling(df, text_column="text", config=lda_config)
print("Top words for topic 0:", lda_result.top_words_by_topic()[0])
```

## Pipeline Stages

- **Preprocessing**: Strip URLs/hashtags/emoji and tokenize Thai text with `pythainlp` stopword removal.
- **BERTopic modeling**: SentenceTransformer embeddings (`paraphrase-multilingual-MiniLM-L12-v2`), `CountVectorizer` Thai unigrams/bigrams, BERTopic with probabilities and optional topic reduction.
- **LDA topic modeling**: CountVectorizer with configurable n-grams/df thresholds feeding `sklearn` LDA for a lightweight alternative.
- **Postprocessing**: Drop noise cluster `-1` (BERTopic), detect topics containing gender/sexism cues, and sample examples per bias topic for annotation.

Adjust knobs via `TopicPipelineConfig` (embedding model, topic size, reduction target, sample size, seed). Bias keywords can be overridden when calling `run_pipeline`.
