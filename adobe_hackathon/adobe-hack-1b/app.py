import os
import json
import re
import fitz  # PyMuPDF
from datetime import datetime
from langdetect import detect
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import time


INPUT_DIR = "input"
OUTPUT_DIR = "output"

BOOST_KEYWORDS = {
    "method", "methodology", "dataset", "data", "evaluation",
    "benchmark", "result", "approach", "architecture", "accuracy"
}

business_keywords = [
    "revenue", "income", "profit", "loss", "earnings", "financial report",
    "research and development", "r&d", "investment", "funding", "capital",
    "market share", "growth", "strategy", "positioning", "competitor",
    "q1", "q2", "q3", "q4", "annual", "2023", "2022", "fiscal"
]

financial_keywords = [
    "revenue", "R&D", "research and development", "investment", "funding", "capital",
    "expenses", "profit", "net income", "earnings", "loss", "cost", "market", "competition",
    "strategy", "positioning", "growth", "trend", "Q1", "Q2", "Q3", "Q4", "2022", "2023", "2024",
    "financial performance", "annual report", "income statement", "balance sheet"
]


def load_text(file_name):
    path = os.path.join(INPUT_DIR, file_name)
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def preprocess_text(text, language):
    return text  # Skip Japanese-specific tokenization


def extract_sections(pdf_path):
    sections = []
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b["type"] != 0:
                continue
            for line in b["lines"]:
                line_text = " ".join([span["text"] for span in line["spans"]]).strip()
                font_sizes = [span["size"] for span in line["spans"]]
                fonts = [span["font"] for span in line["spans"]]

                if not line_text or len(line_text) < 30:
                    continue
                if any(x in line_text.lower() for x in [".py", ".keras", "│", "├──", "└──"]):
                    continue

                try:
                    language = detect(line_text)
                except:
                    language = "unknown"

                processed = preprocess_text(line_text, language)

                sections.append({
                    "document": os.path.basename(pdf_path),
                    "page": page_num,
                    "section_title": line_text[:120],
                    "refined_text": processed[:600],
                    "level": "H1",
                    "language": language,
                    "bold": any("Bold" in font for font in fonts)
                })
    return sections


def boost_score_with_keywords(text, original_score):
    boost = 0
    for word in financial_keywords:
        if word.lower() in text.lower():
            boost += 1
    return original_score + (0.05 * boost)


def score_sections(sections, persona_text, model):
    persona_embedding = model.encode([persona_text])
    for section in sections:
        section_text = section.get("refined_text") or section.get("section_title", "")
        section_embedding = model.encode([section_text])
        sim_score = cosine_similarity([persona_embedding[0]], [section_embedding[0]])[0][0]
        boosted_score = boost_score_with_keywords(section_text, sim_score)
        section["boosted_score"] = boosted_score
    sections.sort(key=lambda x: x["boosted_score"], reverse=True)
    return sections


def main():
    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    persona_text = load_text("persona.txt")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    all_sections = []
    for fname in os.listdir(INPUT_DIR):
        if fname.lower().endswith(".pdf"):
            full_path = os.path.join(INPUT_DIR, fname)
            all_sections.extend(extract_sections(full_path))

    ranked = score_sections(all_sections, persona_text, model)[:20]

    for rank, sec in enumerate(ranked, start=1):
        sec["importance_rank"] = rank

    output = {
        "metadata": {
            "input_documents": [f for f in os.listdir(INPUT_DIR) if f.endswith(".pdf")],
            "persona": persona_text,
            "job_to_be_done": persona_text.split("Job-to-be-done:")[-1].strip() if "Job-to-be-done:" in persona_text else "",
            "processing_timestamp": datetime.utcnow().isoformat() + "Z"
        },
        "extracted_sections": [
            {k: v for k, v in sec.items() if k != "refined_text"}
            for sec in ranked
        ],
        "subsection_analysis": [
            {
                "document": sec["document"],
                "page": sec["page"],
                "refined_text": sec["refined_text"],
                "page_number": sec["page"]
            }
            for sec in ranked
        ]
    }

    with open(os.path.join(OUTPUT_DIR, "challenge1b_output.json"), "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("✅ Output written to challenge1b_output.json")
    print("🕒 Execution Time: %.2f seconds" % (time.time() - start_time))


if __name__ == "__main__":
    main()
