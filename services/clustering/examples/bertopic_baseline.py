# ===============================================================
# 0. CONFIG
# ===============================================================
# ROOT_PATH = "/content/drive/MyDrive/Senior-Project"

import pandas as pd
import numpy as np
import re
import ast

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired

thai_re = re.compile(r"[\u0E00-\u0E7F]")

ROOT_PATH = "services/scraper/output"
# ===============================================================
# 1. LOAD DATA
# ===============================================================
df = pd.read_csv(f"{ROOT_PATH}/postprocessed_output.csv")
df = df.head(1000)

raw_text_list = df["text"].tolist()
df["tokens"] = df["tokens"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
token_text_list = df["tokens"].apply(lambda x: " ".join(x) if isinstance(x, list) else x).tolist()


# ===============================================================
# 2. VERY SIMPLE TOKENIZER (used only for cleaning)
# ===============================================================
def clean_thai(text):
    return " ".join([t for t in text.split() if thai_re.search(t)])


# ===============================================================
# 3. LOAD NOMIC EMBEDDINGS
# ===============================================================
print("Loading Nomic embedding model...")
embedding_model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v2-moe",
    trust_remote_code=True
)

print("Encoding embeddings...")
embeddings = embedding_model.encode(
    raw_text_list,
    show_progress_bar=True,
    convert_to_numpy=True
)


# ===============================================================
# 4. CLEAN DOCS (remove docs that have no Thai at all)
# ===============================================================
clean_docs = [clean_thai(doc) for doc in token_text_list]
valid_idx = [i for i, d in enumerate(clean_docs) if d.strip() != ""]

clean_docs = [clean_docs[i] for i in valid_idx]
filtered_embeddings = embeddings[valid_idx]

print("Docs after cleaning:", len(clean_docs))


# ===============================================================
# 5. TOPIC MODEL (NO COUNT VECTORIZER)
# ===============================================================
rep_model = KeyBERTInspired()  # <-- WORKS WITH THAI!!

topic_model = BERTopic(
    vectorizer_model=None,     # <-- DISABLE COUNT VECTORIZER
    ctfidf_model=None,         # <-- DISABLE C-TFIDF
    representation_model=rep_model,
    verbose=True
)

topics, probs = topic_model.fit_transform(
    clean_docs,
    embeddings=filtered_embeddings
)


# ===============================================================
# 6. BEAUTIFUL THAI LABELS
# ===============================================================
topic_labels = topic_model.generate_topic_labels(
    nr_words=3,
    topic_prefix=False,
    separator=" / "
)
topic_model.set_topic_labels(topic_labels)

topic_info = topic_model.get_topic_info()
print(topic_info.head(10))


# ===============================================================
# 7. OPTIONAL VISUALIZATION
# ===============================================================
try:
    fig = topic_model.visualize_barchart(custom_labels=True)
    fig.show()
except:
    pass