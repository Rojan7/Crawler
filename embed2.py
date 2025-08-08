import os
import json
import numpy as np
import faiss
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# Paths
DATA_DIR = "wikipedia_scrape"
TEXT_META_DIR = os.path.join(DATA_DIR, "meta")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
INDEX_DIR = "faiss_indices"
os.makedirs(INDEX_DIR, exist_ok=True)

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

bge_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
bge_model.to(device)

# Normalize function
def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm

# Embedding functions
def embed_text_bge(text):
    emb = bge_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return emb

def embed_image_clip(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    return normalize(emb[0].cpu().numpy())

# Containers
text_embeddings = []
text_texts = []
text_metadata = []

image_embeddings = []
image_metadata = []

# Process metadata files
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

            emb = embed_text_bge(text)
            text_embeddings.append(emb)
            text_texts.append(text)
            text_metadata.append({
                "page_id": page_id,
                "title": title,
                "url": url,
                "type": "text",
                "section": section,
                "text": text
            })

        elif block_type == "image":
            filename = block.get("filename")
            caption = block.get("caption", "No caption")
            img_path = os.path.join(IMAGE_DIR, filename)

            if not os.path.exists(img_path):
                print(f"[!] Missing image: {img_path}")
                continue

            emb = embed_image_clip(img_path)
            image_embeddings.append(emb)
            image_metadata.append({
                "page_id": page_id,
                "title": title,
                "url": url,
                "type": "image",
                "section": section,
                "filename": filename,
                "caption": caption
            })

print(f"Embedded {len(text_embeddings)} text blocks, {len(image_embeddings)} images.")

# Convert to numpy
text_embeddings = np.array(text_embeddings).astype("float32")
image_embeddings = np.array(image_embeddings).astype("float32")

# Build FAISS indices
text_index = faiss.IndexFlatIP(text_embeddings.shape[1])
text_index.add(text_embeddings)

image_index = faiss.IndexFlatIP(image_embeddings.shape[1])
image_index.add(image_embeddings)

# Save FAISS indices
faiss.write_index(text_index, os.path.join(INDEX_DIR, "text.index"))
faiss.write_index(image_index, os.path.join(INDEX_DIR, "image.index"))

# Save metadata
with open(os.path.join(INDEX_DIR, "text_meta.json"), "w", encoding="utf-8") as f:
    json.dump(text_metadata, f, indent=2, ensure_ascii=False)

with open(os.path.join(INDEX_DIR, "image_meta.json"), "w", encoding="utf-8") as f:
    json.dump(image_metadata, f, indent=2, ensure_ascii=False)

# Optional: Save TF-IDF vectorizer
print("Training TF-IDF vectorizer...")
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(text_texts)

from joblib import dump
dump(tfidf, os.path.join(INDEX_DIR, "tfidf_vectorizer.joblib"))
dump(tfidf_matrix, os.path.join(INDEX_DIR, "tfidf_matrix.joblib"))

print("All embeddings and models saved.")
