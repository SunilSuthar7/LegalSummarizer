# scripts/tokenize_ilc_sample.py

import sys
import json
from pathlib import Path

# 🔧 Add the project root directory to sys.path to allow importing from src/
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.tokenizer import word_tokenize_nltk, remove_punctuation

# 📁 Paths
INPUT_PATH = Path("data/sample_cleaned_ilc.json")
OUTPUT_PATH = Path("data/tokenized_ilc.json")

def load_cleaned_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_tokenized_data(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def tokenize_record(record):
    text = record.get("input_text", "")
    tokens = word_tokenize_nltk(text)
    tokens = remove_punctuation(tokens)
    record["tokenized_text"] = tokens
    return record

def main():
    # 🚀 Load data
    data = load_cleaned_data(INPUT_PATH)
    print(f"📥 Loaded {len(data)} records from {INPUT_PATH}")

    # 🧠 Tokenize each record
    tokenized_data = [tokenize_record(record) for record in data]

    # 🔍 Show a sample
    print("\n🔹 Sample Tokenized Entry:")
    print(json.dumps(tokenized_data[0], indent=2, ensure_ascii=False))

    # 📊 Token count stats
    token_counts = [len(r["tokenized_text"]) for r in tokenized_data]
    print(f"\n📊 Average tokens per document: {sum(token_counts) / len(token_counts):.2f}")
    print(f"📉 Min: {min(token_counts)} | 📈 Max: {max(token_counts)}")

    # 💾 Save output
    save_tokenized_data(tokenized_data, OUTPUT_PATH)
    print(f"\n✅ Saved tokenized data to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
