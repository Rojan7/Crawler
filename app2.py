import streamlit as st
import faiss
import json
import numpy as np
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

# Load FAISS indices and metadata
TEXT_INDEX = faiss.read_index("faiss_indices/text.index")
IMAGE_INDEX = faiss.read_index("faiss_indices/image.index")

with open("faiss_indices/text_meta.json", "r", encoding="utf-8") as f:
    text_meta = json.load(f)

with open("faiss_indices/image_meta.json", "r", encoding="utf-8") as f:
    image_meta = json.load(f)

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"

bge_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
bge_model.to(device)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_text_bge(text):
    emb = bge_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return emb.astype("float32")

def search_text(query, top_k=10):
    q_emb = embed_text_bge(query)
    scores, indices = TEXT_INDEX.search(np.array([q_emb]), top_k)
    results = [text_meta[i] for i in indices[0]]
    return results

def embed_text_clip(text):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        emb = clip_model.get_text_features(**inputs)
    return emb[0].cpu().numpy().astype("float32") / np.linalg.norm(emb[0].cpu().numpy())

def search_images(query, top_k=10):
    q_emb = embed_text_clip(query)
    scores, indices = IMAGE_INDEX.search(np.array([q_emb]), top_k)
    results = [image_meta[i] for i in indices[0]]
    return results

# --- Streamlit UI ---
st.set_page_config(page_title="Multimodal Wikipedia Search", layout="wide")

# CSS for cleaner Google-like look
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    .search-box input {
        font-size: 20px !important;
        padding: 10px;
    }
    .result-card {
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #eee;
    }
    .result-title {
        font-size: 18px;
        color: #1a0dab;
    }
    .result-snippet {
        font-size: 15px;
        color: #4d5156;
    }
    .image-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
    }
    .image-box {
        width: 200px;
    }
    .image-box img {
        width: 100%;
        border-radius: 8px;
    }
    .image-caption {
        font-size: 14px;
        text-align: center;
        margin-top: 4px;
        color: #555;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='font-weight: 400;'>🔍 Multimodal Wikipedia Search</h1>", unsafe_allow_html=True)

# Search bar
query = st.text_input("", placeholder="Search Wikipedia...", label_visibility="collapsed")

if query.strip():
    tab1, tab2 = st.tabs(["📄 All", "🖼️ Images"])

    # Text tab (like Google "All")
    with tab1:
        text_results = search_text(query)
        if text_results:
            for result in text_results:
                st.markdown(f"""
                <div class="result-card">
                    <a class="result-title" href="{result['url']}" target="_blank">{result['title']}</a>
                    <div class="result-snippet">{result.get('text', '')[:300]}...</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.write("No text results found.")

    # Image tab (like Google "Images")
    with tab2:
        image_results = search_images(query)
        if image_results:
            cols = st.columns(4)  # Display 4 images per row
            for i, result in enumerate(image_results):
                img_path = f"wikipedia_scrape/images/{result['filename']}"
                try:
                    img = Image.open(img_path)
                    with cols[i % 4]:
                        st.image(img, use_container_width=True, caption=result.get("caption", ""))
                        st.markdown(f"[{result['title']}]({result['url']})", unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Image not found: {img_path}")
        else:
            st.write("No image results found.")
