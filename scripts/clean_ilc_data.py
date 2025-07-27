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

# Clean all entries
cleaned = []

for idx in tqdm(range(len(train_split)), desc="🧼 Cleaning full ILC dataset"):
    raw = train_split[idx]

    raw_input = raw.get("Case", "")
    raw_summary = raw.get("Summary", "")

    cleaned.append({
        "id": idx,
        "input_text": clean_text(raw_input),
        "summary_text": clean_text(raw_summary)
    })

# Save only JSON (no CSV)
with open("data/sample_cleaned_ilc.json", "w", encoding="utf-8") as f:
    json.dump(cleaned, f, indent=2, ensure_ascii=False)

print("✅ Cleaned ILC dataset saved to → data/sample_cleaned_ilc.json")
