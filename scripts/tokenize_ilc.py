import sys
import os
import json

# Add root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tokenizer import tokenize_text
from tqdm import tqdm

# Paths
INPUT_FILE = "data/sample_cleaned_ilc.json"   # full cleaned ILC dataset
OUTPUT_FILE = "data/tokenized_ilc.json"

# Load cleaned data
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    records = json.load(f)

print(f"📥 Loaded {len(records)} records from {INPUT_FILE}\n")

# Tokenize
tokenized_data = []
token_counts = []

print("🔠 Tokenizing ILC:")
for entry in tqdm(records, desc="Tokenizing ILC"):
    tokens = tokenize_text(entry["input_text"])
    tokenized_data.append({
        "id": entry["id"],
        "tokens": tokens
    })
    token_counts.append(len(tokens))

# Save output
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(tokenized_data, f, indent=2, ensure_ascii=False)

print(f"\n✅ Tokenized full ILC data saved to → {OUTPUT_FILE}")
print(f"📊 Avg tokens/document: {sum(token_counts) / len(token_counts):.2f}")
print(f"📉 Min: {min(token_counts)} | 📈 Max: {max(token_counts)}")
