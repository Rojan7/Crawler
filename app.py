import os
import json
import numpy as np
import faiss
import torch
import streamlit as st
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Paths
TEXT_INDEX_PATH = "faiss_indices/text.index"
TEXT_META_PATH = "faiss_indices/text_meta.json"
IMAGE_INDEX_PATH = "faiss_indices/image.index"
IMAGE_META_PATH = "faiss_indices/image_meta.json"
IMAGE_DIR = "wikipedia_scrape/images"

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

@st.cache_resource
def load_indices_and_metadata():
    text_index = faiss.read_index(TEXT_INDEX_PATH)
    image_index = faiss.read_index(IMAGE_INDEX_PATH)

    with open(TEXT_META_PATH, "r", encoding="utf-8") as f:
        text_metadata = json.load(f)

    with open(IMAGE_META_PATH, "r", encoding="utf-8") as f:
        image_metadata = json.load(f)

    return text_index, text_metadata, image_index, image_metadata

def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def embed_text(text):
    inputs = processor(text=[text], return_tensors="pt", truncation=True, max_length=77, padding="max_length").to(device)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    return normalize(emb[0].cpu().numpy())

def search_faiss(index, metadata, query_vec, top_k=100):
    query_vec = normalize(query_vec)
    D, I = index.search(np.array([query_vec]).astype("float32"), top_k)

    results = []
    query_lower = query.lower()

    for dist, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        entry = metadata[idx]

        searchable_text = (entry.get("title","") + " " + entry.get("section","") + " " + entry.get("text","")).lower()
        keyword_score = 1.0 if query_lower in searchable_text else 0.0

        combined_score = dist + keyword_score  # simple combined scoring

        results.append((entry, combined_score))

    results = sorted(results, key=lambda x: x[1], reverse=True)
    return results[:min(10, len(results))]

# --- Streamlit UI ---
st.set_page_config(page_title="Multimodal Wikipedia Search", layout="wide")
st.title("🔍 Multimodal Wikipedia Search Engine")

text_index, text_metadata, image_index, image_metadata = load_indices_and_metadata()

query = st.text_input("Enter your search query:")

if query:
    query_vec = embed_text(query)

    tab1, tab2 = st.tabs(["📄 Text Results", "🖼️ Image Results"])

    with tab1:
        st.subheader("Text Results")
        text_results = search_faiss(text_index, text_metadata, query_vec, top_k=100)
        if not text_results:
            st.info("No text results found.")
        for res, score in text_results:
            st.markdown(f"### [{res['title']}]({res['url']})")
            st.markdown(f"**Section:** {res.get('section', '')}")
            st.write(res.get("text", "")[:300] + "...")
            st.markdown("---")

    with tab2:
        st.subheader("Image Results")
        image_results = search_faiss(image_index, image_metadata, query_vec, top_k=100)
        if not image_results:
            st.info("No image results found.")
        cols = st.columns(3)
        for i, (res, score) in enumerate(image_results):
            with cols[i % 3]:
                img_path = os.path.join(IMAGE_DIR, res["filename"])
                if os.path.exists(img_path):
                    st.image(img_path, caption=res.get("caption", "No caption"), use_container_width=True)
                st.markdown(f"[{res['title']}]({res['url']})")
