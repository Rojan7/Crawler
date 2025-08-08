import os
import json
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# Paths
DATA_DIR = "wikipedia_scrape"
TEXT_META_DIR = os.path.join(DATA_DIR, "meta")
IMAGE_DIR = os.path.join(DATA_DIR, "images")

INDEX_DIR = "faiss_indices_bge"
os.makedirs(INDEX_DIR, exist_ok=True)

# Load BGE text model
text_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm

def embed_text(text):
    emb = text_model.encode(text, normalize_embeddings=True)
    return emb

# Containers for embeddings and metadata
text_embeddings = []
text_metadata = []
raw_texts = []

# Read all JSON metadata files
meta_files = [f for f in os.listdir(TEXT_META_DIR) if f.endswith(".json")]
print(f"Found {len(meta_files)} metadata files.")

for meta_file in meta_files:
    with open(os.path.join(TEXT_META_DIR, meta_file), "r", encoding="utf-8") as f:
        page_meta = json.load(f)

    page_id = page_meta["page_id"]
    title = page_meta["title"]
    url = page_meta["url"]

    for block in page_meta["content"]:
        block_type = block["type"]
        section = block.get("section", "")

        if block_type == "text":
            text = block.get("content", "").strip()
            if not text:
                continue

            emb = embed_text(text)
            text_embeddings.append(emb)
            raw_texts.append(text)

            text_metadata.append({
                "page_id": page_id,
                "title": title,
                "url": url,
                "type": "text",
                "section": section,
                "text": text,
            })

print(f"Embedded {len(text_embeddings)} text blocks.")

# TF-IDF fallback index
print("Training TF-IDF vectorizer...")
tfidf_vectorizer = TfidfVectorizer(max_features=50000)
tfidf_matrix = tfidf_vectorizer.fit_transform(raw_texts)

# Save TF-IDF vectorizer and matrix
from joblib import dump

dump(tfidf_vectorizer, os.path.join(INDEX_DIR, "tfidf_vectorizer.joblib"))
np.save(os.path.join(INDEX_DIR, "tfidf_matrix.npy"), tfidf_matrix.toarray())

# Convert to numpy and build FAISS index
text_embeddings = np.array(text_embeddings).astype("float32")
text_index = faiss.IndexFlatIP(text_embeddings.shape[1])
text_index.add(text_embeddings)

faiss.write_index(text_index, os.path.join(INDEX_DIR, "text_bge.index"))
with open(os.path.join(INDEX_DIR, "text_meta.json"), "w", encoding="utf-8") as f:
    json.dump(text_metadata, f, indent=2, ensure_ascii=False)

print("Text embeddings, FAISS index, and TF-IDF saved.")