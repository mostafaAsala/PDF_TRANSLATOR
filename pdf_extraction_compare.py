import os
import time
import fitz  # PyMuPDF
import pdfplumber
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBoxHorizontal, LTTextLineHorizontal

PDF_PATH = r"C:\Users\161070\Downloads\data1.pdf"  # Change this to your PDF path
OUTPUT_DIR = "extracted_text"



def extract_with_pymupdf_spans(path):
    doc = fitz.open(path)
    items = []
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text:
                            items.append({
                                "text": f'"{text}"',
                                "bbox": tuple(span["bbox"])
                            })
    return items


def extract_with_pdfplumber(path):
    items = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            for word in page.extract_words():
                items.append({
                    "text": f'"{word["text"]}"',
                    "bbox": (word["x0"], word["top"], word["x1"], word["bottom"])
                })
    return items


def extract_with_pdfminer(path):
    items = []
    for page_layout in extract_pages(path):
        for element in page_layout:
            if isinstance(element, (LTTextBoxHorizontal, LTTextLineHorizontal)):
                bbox = element.bbox
                text = element.get_text().strip()
                if text:
                    items.append({"text": f'"{text}"', "bbox": bbox})
    return items


def save_results_to_file(method_name, items, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{method_name.lower()}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(f"{item['text']} | BBox: {item['bbox']}\n")


methods = {
    "PyMuPDF": extract_with_pymupdf_spans,
    "pdfplumber": extract_with_pdfplumber,
    "pdfminer": extract_with_pdfminer,
}

print("📄 PDF Text + BBox Extraction + Save to File\n")

for name, func in methods.items():
    start = time.time()
    try:
        items = func(PDF_PATH)
        elapsed = time.time() - start
        save_results_to_file(name, items, OUTPUT_DIR)
        print(f"✅ {name} | Time: {elapsed:.3f} sec | Items: {len(items)} | Saved to {name.lower()}.txt")
    except Exception as e:
        print(f"❌ {name} failed: {e}")

print("\n📁 All results saved in:", os.path.abspath(OUTPUT_DIR))
