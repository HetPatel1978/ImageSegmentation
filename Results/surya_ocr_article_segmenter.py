# surya_ocr_article_segmenter.py

import json
import os

def load_ocr_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def sort_blocks(blocks):
    return sorted(blocks, key=lambda b: (b['bbox'][1], b['bbox'][0]))

def vertical_overlap(b1, b2, threshold=30):
    return abs(b1['bbox'][1] - b2['bbox'][3]) < threshold

def cluster_blocks(sorted_blocks, vertical_threshold=30):
    clusters = []
    current_cluster = [sorted_blocks[0]]

    for i in range(1, len(sorted_blocks)):
        prev = current_cluster[-1]
        curr = sorted_blocks[i]

        if vertical_overlap(prev, curr, threshold=vertical_threshold):
            current_cluster.append(curr)
        else:
            clusters.append(current_cluster)
            current_cluster = [curr]

    if current_cluster:
        clusters.append(current_cluster)

    return clusters

def extract_articles_from_clusters(clusters):
    articles = []

    for cluster in clusters:
        title = ""
        paragraphs = []

        for block in cluster:
            text = block.get("text", "").strip()
            if not text:
                continue

            if not title and block['label'] in ["SectionHeader", "PageHeader"]:
                title = text
            elif not title:
                title = text  # Fallback if no header found
            else:
                paragraphs.append(text)

        if title:
            articles.append({"title": title, "paragraphs": paragraphs})

    return articles

def export_to_json(articles, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"articles": articles}, f, indent=4, ensure_ascii=False)

def export_to_markdown(articles, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for article in articles:
            f.write(f"# {article['title']}\n\n")
            for para in article['paragraphs']:
                f.write(f"{para}\n\n")

if __name__ == "__main__":
    input_json = "ocr_results.json"  # Update path if needed
    output_json = "articles_output.json"
    output_md = "articles_output.md"

    print("Loading OCR data...")
    raw_data = load_ocr_data(input_json)

    print("Combining text lines from all pages...")
    data = []
    for page_key, page_value in raw_data.items():
        if isinstance(page_value, dict) and "text_lines" in page_value:
            for item in page_value["text_lines"]:
                text = item.get("text", "").strip()
                label = item.get("type", "Text")  # fallback to 'Text' if label missing
                # Convert polygon to bbox if bbox is missing
                if "bbox" not in item and "polygon" in item:
                    x_coords = [pt[0] for pt in item["polygon"]]
                    y_coords = [pt[1] for pt in item["polygon"]]
                    item["bbox"] = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                data.append({
                    "text": text,
                    "label": label,
                    "bbox": item.get("bbox")
                })

    print(f"Loaded {len(data)} text elements from all pages.")

    print("Filtering relevant blocks...")
    text_blocks = [
        {
            "text": block["text"],
            "label": block["label"],
            "bbox": block["bbox"]
        }
        for block in data
        if block["label"] in ["Text", "SectionHeader", "PageHeader"]
    ]

    print("Sorting blocks top-down, left-right...")
    sorted_blocks = sort_blocks(text_blocks)

    print("Clustering blocks into articles...")
    clusters = cluster_blocks(sorted_blocks)

    print("Extracting articles and paragraphs...")
    articles = extract_articles_from_clusters(clusters)

    print(f"Exporting to {output_json} and {output_md}...")
    export_to_json(articles, output_json)
    export_to_markdown(articles, output_md)

    print("Done! ✅")