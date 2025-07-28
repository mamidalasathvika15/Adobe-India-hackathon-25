import fitz  # PyMuPDF
import os
import json
import re
from collections import Counter
from langdetect import detect

INPUT_DIR = "input"
OUTPUT_DIR = "output"

def detect_title(doc):
    """Smarter title detection using metadata + top of first page"""
    meta_title = doc.metadata.get("title")
    if meta_title and len(meta_title.strip()) > 5:
        return meta_title.strip()

    # Fallback: get large bold text from top 25% of first page
    first_page = doc[0]
    blocks = first_page.get_text("dict")["blocks"]
    top_blocks = [b for b in blocks if b.get("type") == 0 and b["bbox"][1] < first_page.rect.height * 0.25]

    for block in top_blocks:
        for line in block["lines"]:
            line_text = " ".join(span["text"] for span in line["spans"]).strip()
            if line_text and len(line_text) > 5:
                return line_text
    return os.path.basename(doc.name)

def classify_heading_level(text):
    """Infer heading level from numbering (e.g., 1., 1.2, 1.2.3 = H1, H2, H3)"""
    match = re.match(r'^(\d+)(\.\d+)*\s+', text)
    if not match:
        return None
    depth = text.count(".")
    return f"H{min(depth + 1, 3)}"

def extract_outline(doc):
    """Extract structured outline with headings based on visual & style features"""
    font_sizes = []
    style_counts = Counter()
    candidates = []

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                full_line_text = ""
                span_fonts = []
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    font_size = round(span["size"], 1)
                    font_name = span.get("font", "default")
                    flags = span.get("flags", 0)
                    is_bold = bool(flags & 2)
                    span_fonts.append((font_size, font_name, is_bold))
                    font_sizes.append(font_size)
                    full_line_text += text + " "
                full_line_text = full_line_text.strip()

                if full_line_text:
                    try:
                        language = detect(full_line_text)
                    except:
                        language = "unknown"

                    most_common = Counter(span_fonts).most_common(1)[0][0]
                    candidates.append({
                        "text": full_line_text,
                        "page": page_num,
                        "font_size": most_common[0],
                        "font_name": most_common[1],
                        "is_bold": most_common[2],
                        "language": language
                    })

    if not font_sizes:
        return []

    # Get body text size (most frequent)
    most_common_font_size = Counter(font_sizes).most_common(1)[0][0]

    outline = []
    seen = set()

    for c in candidates:
        if c["text"] in seen or len(c["text"]) < 5:
            continue
        seen.add(c["text"])

        # Heuristics to detect headings
        is_heading = (
            c["font_size"] > most_common_font_size + 1.0 or
            c["is_bold"] or
            c["text"].isupper() or
            re.match(r'^\d+(\.\d+)*\s+', c["text"])
        )

        if is_heading:
            level = classify_heading_level(c["text"]) or "H1"
            outline.append({
                "level": level,
                "text": c["text"],
                "page": c["page"],
                "language": c["language"]
            })

    return outline

def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    title = detect_title(doc)
    outline = extract_outline(doc)
    return {
        "title": title,
        "outline": outline
    }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for fname in os.listdir(INPUT_DIR):
        if fname.lower().endswith(".pdf"):
            pdf_path = os.path.join(INPUT_DIR, fname)
            result = process_pdf(pdf_path)
            output_file = os.path.splitext(fname)[0] + ".json"
            with open(os.path.join(OUTPUT_DIR, output_file), "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"[✓] Processed {fname} → {output_file}")

if __name__ == "__main__":
    main()
