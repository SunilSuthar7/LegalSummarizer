import sys
import os
import json

# Add root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tokenizer import tokenize_text
from tqdm import tqdm

# Input path
INPUT_FILE = "data/cleaned_inabs.json"
OUTPUT_FILE = "data/tokenized_inabs.json"

# Load cleaned IN-ABS data
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    records = json.load(f)

print(f"📥 Loaded {len(records)} records from {INPUT_FILE}\n")

# Tokenize
tokenized_data = []
token_counts = []

print("🔠 Tokenizing IN-ABS:")
for entry in tqdm(records, desc="Tokenizing IN-ABS"):
    tokens = tokenize_text(entry["input_text"])
    tokenized_data.append({
        "id": entry["id"],
        "tokens": tokens
    })
    token_counts.append(len(tokens))

# Save to output file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(tokenized_data, f, indent=2, ensure_ascii=False)

print(f"\n✅ Tokenized full IN-ABS data saved to → {OUTPUT_FILE}")
print(f"📊 Avg tokens/document: {sum(token_counts) / len(token_counts):.2f}")
print(f"📉 Min: {min(token_counts)} | 📈 Max: {max(token_counts)}")
