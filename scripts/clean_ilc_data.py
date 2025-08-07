import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import load_dataset
from src.cleaner import clean_text
import pandas as pd
from tqdm import tqdm
import json

# Create output directory
os.makedirs("data", exist_ok=True)

# Load full ILC dataset
print("📦 Loading ILC dataset...")
dataset = load_dataset("d0r1h/ILC")
train_split = dataset["train"]

print(f"📊 Total records in ILC: {len(train_split)}")

# Clean all entries using shared cleaner
cleaned = []

for idx in tqdm(range(len(train_split)), desc=" Cleaning full ILC dataset"):
    raw = train_split[idx]
    raw_input = raw.get("Case", "")
    raw_summary = raw.get("Summary", "")

    cleaned_input = clean_text(raw_input, aggressive=False)
    cleaned_summary = clean_text(raw_summary, aggressive=False)

    if cleaned_input.strip() and cleaned_summary.strip():
        cleaned.append({
            "id": idx,
            "input_text": cleaned_input,
            "summary_text": cleaned_summary
        })

# Save JSON output (no CSV)
with open("data/cleaned_ilc.json", "w", encoding="utf-8") as f:
    json.dump(cleaned, f, indent=2, ensure_ascii=False)

print("✅ Cleaned ILC dataset saved to → data/sample_cleaned_ilc.json")
