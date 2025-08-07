import json
import os
from tqdm import tqdm

# Input files
INPUT_FILES = {
    "inabs": "data/cleaned_inabs.json",
    "ilc": "data/cleaned_ilc.json"
}

# Output directory
OUTPUT_DIR = "data/legalbert_ready"

# Preprocessing config
MIN_WORDS = 50
MAX_WORDS = 3000

def clean_text(text):
    """Basic normalization: remove excessive whitespace, truncate"""
    if not isinstance(text, str):
        return ""
    text = ' '.join(text.split())  # Normalize spaces
    return text[:MAX_WORDS]        # Truncate long docs

def prepare_file(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"❌ Skipping {input_path}: File not found.")
        return 0, 0

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ready_docs = []
    skipped = 0

    for doc in tqdm(data, desc=f"Processing {os.path.basename(input_path)}"):
        doc_id = doc.get("id")
        text = clean_text(doc.get("input_text", ""))
        word_count = len(text.split())

        if word_count < MIN_WORDS:
            skipped += 1
            continue

        ready_docs.append({
            "id": doc_id,
            "input_text": text,
            "word_count": word_count
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ready_docs, f, indent=2, ensure_ascii=False)

    return len(ready_docs), skipped

if __name__ == "__main__":
    total = 0
    skipped = 0

    for key, path in INPUT_FILES.items():
        output_path = os.path.join(OUTPUT_DIR, f"{key}_legalbert_ready.json")
        count, skip = prepare_file(path, output_path)
        print(f"✅ {count} documents saved to {output_path}")
        if skip > 0:
            print(f"❌ Skipped {skip} documents due to size/format issues.")
        total += count
        skipped += skip

    print(f"\n🎯 Total processed: {total}, Skipped: {skipped}")
