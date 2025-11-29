import pandas as pd
from clustering import run_pipeline, TopicPipelineConfig

df = pd.read_csv("services/clustering/assets/postprocessed_output.csv")
config = TopicPipelineConfig(
    embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    min_topic_size=30,
    reduce_to=25,
)

result = run_pipeline(df, text_column="text", config=config)
result.save("bertopic_output.csv", model_dir="bertopic_thai_gender_bias")

print("Bias topics:", result.bias_topics)
print("Sampled sentences from the first bias topic:", result.biased_samples.get(result.bias_topics[0], []))