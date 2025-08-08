import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
from PIL import Image
from io import BytesIO
import json
from tqdm import tqdm

# ---- Config ----
start_url = "https://en.wikipedia.org/wiki/Kallang_Field"
allowed_domain = "en.wikipedia.org"
max_pages = 200  # Adjust as needed

output_dir = "wikipedia_scrape"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "meta"), exist_ok=True)

# ---- Counters ----
page_count = 99
image_count = 936

def extract_content(url, page_id, pbar=None):
    global image_count
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        title = soup.title.string.strip() if soup.title else "No Title"

        # Remove scripts/styles
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        content_blocks = []
        current_section = ""

        for tag in soup.find_all(["h1", "h2", "h3", "p", "img"]):
            if tag.name in ["h1", "h2", "h3"]:
                current_section = tag.get_text().strip()
            elif tag.name == "p":
                text = tag.get_text().strip()
                if text:
                    content_blocks.append({
                        "type": "text",
                        "section": current_section,
                        "content": text
                    })
            elif tag.name == "img" and tag.has_attr("src"):
                img_src = tag["src"]
                img_url = urljoin(url, img_src)
                try:
                    img_res = requests.get(img_url, timeout=10)
                    img = Image.open(BytesIO(img_res.content))
                    if img.width < 100 or img.height < 100:
                        continue

                    img_name = f"image_{image_count}.jpg"
                    img_path = os.path.join(output_dir, "images", img_name)
                    img.convert("RGB").save(img_path)

                    caption = tag.get("alt") or tag.get("title") or "No caption"

                    content_blocks.append({
                        "type": "image",
                        "section": current_section,
                        "filename": img_name,
                        "caption": caption
                    })
                    image_count += 1
                except:
                    continue

        # Save metadata JSON
        metadata = {
            "page_id": page_id,
            "url": url,
            "title": title,
            "content": content_blocks
        }

        meta_file = os.path.join(output_dir, "meta", f"meta_{page_id}.json")
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        if pbar:
            pbar.set_description(f"Scraped: {title[:50]}")
            pbar.update(1)
        else:
            print(f"[✓] Scraped: {title} ({url})")

    except Exception as e:
        print(f"[!] Failed to scrape {url}: {e}")

def extract_links(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        links = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            full_url = urljoin(url, href)
            parsed = urlparse(full_url)

            if parsed.netloc == allowed_domain and parsed.scheme.startswith("http"):
                if "/wiki/" in parsed.path and ":" not in parsed.path.split("/wiki/")[-1]:
                    links.add(full_url)
        return links
    except:
        return set()

def bfs_crawl(start_url, max_pages):
    visited = set()
    queue = deque([start_url])
    global page_count

    with tqdm(total=max_pages, desc="Crawling Wikipedia") as pbar:
        while queue and page_count < max_pages:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            extract_content(current, page_count, pbar)
            page_count += 1

            for link in extract_links(current):
                if link not in visited:
                    queue.append(link)

if __name__ == "__main__":
    bfs_crawl(start_url, max_pages)
