import dash
from dash import html, dcc, Input, Output, State
import base64
from PIL import Image
import io
import numpy as np
import faiss
import json
import torch
from transformers import CLIPProcessor, CLIPModel

INDEX_DIR = "indices1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model and processor
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

app = dash.Dash(__name__)

app.layout = html.Div(
    style={
        'maxWidth': '600px',
        'margin': 'auto',
        'padding': '40px 20px',
        'fontFamily': 'Arial, sans-serif',
        'textAlign': 'center',
    },
    children=[
        # Google logo
        html.Img(
            src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTUN4Axu0W4A5AdoBXnMk_9A0LtyW7bPU1HKQ&s",
            style={'marginBottom': '30px', 'width': '272px', 'height': '92px'}
        ),

        dcc.Input(
            id='text-query',
            type='text',
            placeholder='Search or type a URL',
            style={
                'width': '100%',
                'height': '44px',
                'fontSize': '18px',
                'padding': '0 15px',
                'border': '1px solid #dfe1e5',
                'borderRadius': '24px',
                'boxShadow': 'none',
                'outline': 'none',
                'transition': 'box-shadow 0.2s ease-in-out',
                'marginBottom': '12px',
            }
        ),

        dcc.Upload(
            id='upload-image',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select an Image')
            ]),
            style={
                'width': '100%',
                'height': '40px',
                'lineHeight': '40px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '24px',
                'textAlign': 'center',
                'marginBottom': '30px',
                'color': '#555',
                'cursor': 'pointer',
                'fontSize': '14px',
            },
            accept='image/*',
            multiple=False
        ),

        html.Div(
            style={'display': 'flex', 'justifyContent': 'center', 'gap': '12px'},
            children=[
                html.Button(
                    'Search',
                    id='search-button',
                    n_clicks=0,
                    style={
                        'backgroundColor': '#f8f9fa',
                        'border': '1px solid #f8f9fa',
                        'color': '#3c4043',
                        'fontSize': '14px',
                        'padding': '10px 20px',
                        'borderRadius': '4px',
                        'cursor': 'pointer',
                        'boxShadow': '0 1px 1px rgb(0 0 0 / 0.1)',
                        'userSelect': 'none',
                        'fontWeight': 'bold',
                        'fontFamily': "'Arial', sans-serif",
                    }
                ),
                html.Button(
                    "Reset",
                    n_clicks=0,
                    style={
                        'backgroundColor': '#f8f9fa',
                        'border': '1px solid #f8f9fa',
                        'color': '#3c4043',
                        'fontSize': '14px',
                        'padding': '10px 20px',
                        'borderRadius': '4px',
                        'cursor': 'pointer',
                        'boxShadow': '0 1px 1px rgb(0 0 0 / 0.1)',
                        'userSelect': 'none',
                        'fontWeight': 'bold',
                        'fontFamily': "'Arial', sans-serif",
                    }
                ),
            ]
        ),

        html.Div(id='results', style={'marginTop': '40px', 'textAlign': 'left'}),
    ]
)

def parse_uploaded_image(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return Image.open(io.BytesIO(decoded)).convert('RGB')

@app.callback(
    Output('results', 'children'),
    Input('search-button', 'n_clicks'),
    State('text-query', 'value'),
    State('upload-image', 'contents'),
)
def search(n_clicks, text_query, uploaded_image_contents):
    if n_clicks == 0:
        return ""

    # Prioritize image search if uploaded
    if uploaded_image_contents:
        image = parse_uploaded_image(uploaded_image_contents)
        query_emb = embed_image_clip_pil(image)
        D_img, I_img = image_index.search(np.array([query_emb]), 5)
        D_text, I_text = text_index.search(np.array([query_emb]), 5)

        # Show the uploaded image on top
        uploaded_img_display = html.Div([
            html.H3("🔎 Search Image"),
            html.Img(
                src=uploaded_image_contents,
                style={'maxWidth': '300px', 'borderRadius': '8px', 'marginBottom': '20px'}
            )
        ])
    elif text_query and text_query.strip():
        query_emb = embed_text_clip(text_query.strip())
        D_text, I_text = text_index.search(np.array([query_emb]), 5)
        D_img, I_img = image_index.search(np.array([query_emb]), 5)
        uploaded_img_display = None
    else:
        return html.Div("Please enter a query or upload an image.", style={'color': '#777', 'fontStyle': 'italic'})

    # Text results
    text_results = []
    for idx in I_text[0]:
        item = text_meta[idx]
        text_results.append(html.Div([
            html.A(item['title'], href=item['url'], target='_blank', style={'color': '#1a0dab', 'fontSize': '18px', 'textDecoration': 'none'}),
            html.P(item.get("text", "")[:300] + "...", style={'color': '#4d5156', 'marginTop': '4px'})
        ], style={'borderBottom': '1px solid #e8e8e8', 'paddingBottom': '10px', 'marginBottom': '20px'}))

    # Image results
    image_results = []
    for idx in I_img[0]:
        item = image_meta[idx]
        img_path = f"wikipedia_scrape/images/{item['filename']}"
        try:
            img = Image.open(img_path)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            image_results.append(html.Div([
                html.Img(src=f"data:image/png;base64,{img_str}", style={'width': '100%', 'height': '180px', 'objectFit': 'cover', 'borderRadius': '5px'}),
                html.A(item['title'], href=item['url'], target='_blank', style={'color': '#1a0dab', 'textDecoration': 'none', 'display': 'block', 'marginTop': '5px'}),
                html.P(item.get("caption", ""), style={'color': '#6b7280', 'fontSize': '14px', 'marginTop': '2px'})
            ], style={'width': '23%', 'margin': '1%', 'boxShadow': '0 1px 3px rgb(0 0 0 / 0.1)', 'borderRadius': '5px', 'padding': '5px', 'display': 'inline-block', 'verticalAlign': 'top'}))
        except Exception:
            pass

    results_children = []
    if uploaded_image_contents:
        results_children.append(uploaded_img_display)

    results_children.append(html.H2("📄 Text Results", style={'fontWeight': '600', 'marginBottom': '15px'}))
    results_children.extend(text_results)

    results_children.append(html.H2("🖼️ Image Results", style={'fontWeight': '600', 'marginTop': '40px', 'marginBottom': '15px'}))
    results_children.append(html.Div(image_results, style={'display': 'flex', 'flexWrap': 'wrap'}))

    return html.Div(results_children)

if __name__ == '__main__':
    app.run(debug=True)
