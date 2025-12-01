# Clustering Service

Thai BERTopic pipeline for gender-bias topic mining. Handles preprocessing, topic modeling, bias-topic detection, and exporting results.

## Quickstart

```bash
cd services/clustering
pip install -e .
python -m clustering --input ../testset.csv --text-column text --output-csv assets/bertopic_output.csv --model-dir assets/bertopic_thai_gender_bias
```

This writes per-document topics to `assets/bertopic_output.csv`, saves the trained model, and prints detected bias topics.

## Programmatic Use

```python
import pandas as pd
from clustering import run_pipeline, TopicPipelineConfig

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
```

## Pipeline Stages

- **Preprocessing**: Strip URLs/hashtags/emoji and tokenize Thai text with `pythainlp` stopword removal.
- **Modeling**: SentenceTransformer embeddings (`paraphrase-multilingual-MiniLM-L12-v2`), `CountVectorizer` Thai unigrams/bigrams, BERTopic with probabilities and optional topic reduction.
- **Postprocessing**: Drop noise cluster `-1`, detect topics containing gender/sexism cues, and sample examples per bias topic for annotation.

Adjust knobs via `TopicPipelineConfig` (embedding model, topic size, reduction target, sample size, seed). Bias keywords can be overridden when calling `run_pipeline`.
