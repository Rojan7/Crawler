import os
import json
import numpy as np
import faiss
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Paths
DATA_DIR = "wikipedia_scrape"
TEXT_META_DIR = os.path.join(DATA_DIR, "meta")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
INDEX_DIR = "indices1"
os.makedirs(INDEX_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model & processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm

def embed_text_clip(text):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        emb = clip_model.get_text_features(**inputs)
    return normalize(emb[0].cpu().numpy()).astype("float32")

def embed_image_clip(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    return normalize(emb[0].cpu().numpy()).astype("float32")

# Containers
text_embeddings = []
text_metadata = []

image_embeddings = []
image_metadata = []

meta_files = [f for f in os.listdir(TEXT_META_DIR) if f.endswith(".json")]
print(f"Processing {len(meta_files)} metadata files...")

for meta_file in meta_files:
    with open(os.path.join(TEXT_META_DIR, meta_file), "r", encoding="utf-8") as f:
        page_meta = json.load(f)

    page_id = page_meta.get("page_id")
    title = page_meta.get("title", "")
    url = page_meta.get("url", "")

    for block in page_meta.get("content", []):
        block_type = block.get("type")
        section = block.get("section", "")

        if block_type == "text":
            text = block.get("content", "").strip()
            if not text:
                continue
            emb = embed_text_clip(text)
            text_embeddings.append(emb)
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
                print(f"[!] Missing image file: {img_path}")
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

print(f"Embedded {len(text_embeddings)} text blocks and {len(image_embeddings)} images.")

text_embeddings = np.array(text_embeddings).astype("float32")
image_embeddings = np.array(image_embeddings).astype("float32")

# Build FAISS indices (512 dim)
dim = 512
text_index = faiss.IndexFlatIP(dim)
text_index.add(text_embeddings)

image_index = faiss.IndexFlatIP(dim)
image_index.add(image_embeddings)

# Save indices and metadata
faiss.write_index(text_index, os.path.join(INDEX_DIR, "text.index"))
faiss.write_index(image_index, os.path.join(INDEX_DIR, "image.index"))

with open(os.path.join(INDEX_DIR, "text_meta.json"), "w", encoding="utf-8") as f:
    json.dump(text_metadata, f, indent=2, ensure_ascii=False)

with open(os.path.join(INDEX_DIR, "image_meta.json"), "w", encoding="utf-8") as f:
    json.dump(image_metadata, f, indent=2, ensure_ascii=False)

print("Embedding and index creation complete.")
