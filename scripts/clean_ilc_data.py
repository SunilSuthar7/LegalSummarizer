from datasets import load_dataset
import pandas as pd
import re
import json
import os
from tqdm import tqdm

# Step 1: Load ILC dataset
print("📦 Loading ILC dataset...")
dataset = load_dataset("d0r1h/ILC")
data = dataset["train"]

# Step 2: Cleaning utilities
def fix_encoding(text):
    try:
        return text.encode("latin1", "ignore").decode("utf-8", "ignore")
    except:
        return text

def clean_whitespace(text):
    return re.sub(r'\s+', ' ', str(text)).strip()

def remove_html_tags(text):
    return re.sub(r'<[^>]+>', '', str(text))

def normalize_punctuation(text):
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('–', '-').replace('•', '-')
    return text

def clean_text(text):
    text = fix_encoding(text)
    text = clean_whitespace(text)
    text = remove_html_tags(text)
    text = normalize_punctuation(text)
    return text

# Step 3: Clean all rows using correct field names
print("🧼 Cleaning...")
cleaned = []

for idx, sample in enumerate(tqdm(data)):
    raw_input = sample.get("Case", "")
    raw_summary = sample.get("Summary", "")

    cleaned_input = clean_text(raw_input)
    cleaned_summary = clean_text(raw_summary)

    if cleaned_input.strip() and cleaned_summary.strip():
        cleaned.append({
            "id": idx,
            "input_text": cleaned_input,
            "summary_text": cleaned_summary
        })

print(f"✅ Cleaned {len(cleaned)} entries.")

# Step 4: Save CSV and JSON
os.makedirs("../data", exist_ok=True)

# Save CSV
df = pd.DataFrame(cleaned)
df.to_csv("../data/sample_cleaned_ilc.csv", index=False)
print("✅ Saved CSV → ../data/sample_cleaned_ilc.csv")

# Save JSON
with open("../data/sample_cleaned_ilc.json", "w", encoding="utf-8") as f:
    json.dump(cleaned, f, indent=2, ensure_ascii=False)
print("✅ Saved JSON → ../data/sample_cleaned_ilc.json")

# Step 5: Show samples
print("\n🔍 Cleaned Sample Preview:")
for i in range(min(3, len(cleaned))):
    print(f"\n📄 Sample {i+1}:\nID: {cleaned[i]['id']}")
    print(f"Input: {cleaned[i]['input_text'][:300]}...")
    print(f"→ Summary: {cleaned[i]['summary_text'][:150]}...")
