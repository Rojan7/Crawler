import streamlit as st
import faiss
import json
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import base64
from io import BytesIO

INDEX_DIR = "indices1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load FAISS indices and metadata
text_index = faiss.read_index(f"{INDEX_DIR}/text.index")
image_index = faiss.read_index(f"{INDEX_DIR}/image.index")

with open(f"{INDEX_DIR}/text_meta.json", "r", encoding="utf-8") as f:
    text_meta = json.load(f)

with open(f"{INDEX_DIR}/image_meta.json", "r", encoding="utf-8") as f:
    image_meta = json.load(f)

def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm

def embed_text_clip(text):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        emb = clip_model.get_text_features(**inputs)
    return normalize(emb[0].cpu().numpy()).astype("float32")

def embed_image_clip_pil(image):
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    return normalize(emb[0].cpu().numpy()).astype("float32")

# Primitive CSS override for clean white UI
st.markdown("""
<style>
body, .main, .block-container {
    background-color: #fff !important;
    color: #202124;
    font-family: Arial, sans-serif;
}
.block-container {
    max-width: 900px;
    margin: 2rem auto;
    padding: 0 1rem;
}
.stTextInput > div > div > input {
    width: 100% !important;
    max-width: 600px;
    height: 44px;
    font-size: 18px !important;
    border: 1px solid #dfe1e5;
    border-radius: 24px;
    padding: 0 20px;
    box-shadow: none;
    transition: box-shadow 0.2s ease;
}
.stTextInput > div > div > input:focus {
    border-color: #4285f4;
    box-shadow: 0 1px 6px rgba(66, 133, 244, 0.6);
    outline: none;
}
h1, h2 {
    font-weight: 400;
    color: #202124;
}
.result-card {
    margin-bottom: 1.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid #e8e8e8;
}
.result-title {
    font-size: 20px;
    color: #1a0dab;
    text-decoration: none;
    cursor: pointer;
}
.result-title:hover {
    text-decoration: underline;
}
.result-snippet {
    font-size: 14px;
    color: #4d5156;
    margin-top: 4px;
    line-height: 1.4;
}
.image-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
}
.image-card {
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 4px rgb(0 0 0 / 0.1);
}
.image-card img {
    width: 100%;
    height: 120px;
    object-fit: cover;
}
.image-caption {
    padding: 0.5rem;
    font-size: 0.85rem;
    color: #5f6368;
}
.image-caption a {
    color: #1a0dab;
    text-decoration: none;
}
.image-caption a:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1>🔍 Multimodal Wikipedia Search</h1>', unsafe_allow_html=True)
st.markdown('<p>Search by text or upload an image. Results show text and images side-by-side.</p>', unsafe_allow_html=True)

query_text = st.text_input("", placeholder="Enter search query (text):")

uploaded_image = st.file_uploader("Or upload an image for search", type=["png", "jpg", "jpeg"])

if query_text.strip() or uploaded_image:
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        query_emb_img = embed_image_clip_pil(image)
        D_img, I_img = image_index.search(np.array([query_emb_img]), 5)
        D_text, I_text = text_index.search(np.array([query_emb_img]), 5)
    else:
        query_emb_text = embed_text_clip(query_text)
        D_text, I_text = text_index.search(np.array([query_emb_text]), 5)
        D_img, I_img = image_index.search(np.array([query_emb_text]), 5)

    st.markdown('<h2>📄 Text Results</h2>', unsafe_allow_html=True)
    for idx in I_text[0]:
        item = text_meta[idx]
        html = f'''
        <div class="result-card">
            <a href="{item['url']}" target="_blank" class="result-title">{item['title']}</a>
            <p class="result-snippet">{item.get("text", "")[:300]}...</p>
        </div>
        '''
        st.markdown(html, unsafe_allow_html=True)

    st.markdown('<h2>🖼️ Image Results</h2>', unsafe_allow_html=True)
    st.markdown('<div class="image-grid">', unsafe_allow_html=True)
    for idx in I_img[0]:
        item = image_meta[idx]
        img_path = f"wikipedia_scrape/images/{item['filename']}"
        try:
            img = Image.open(img_path)
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            image_html = f'''
            <div class="image-card">
                <img src="data:image/png;base64,{img_str}" alt="{item.get("caption", "")}" />
                <div class="image-caption">
                    <a href="{item['url']}" target="_blank">{item['title']}</a><br/>
                    {item.get("caption", "")}
                </div>
            </div>
            '''
            st.markdown(image_html, unsafe_allow_html=True)
        except Exception:
            st.warning(f"Image not found: {img_path}")
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Enter a text query or upload an image to search.")
