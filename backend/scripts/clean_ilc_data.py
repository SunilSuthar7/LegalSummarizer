import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import load_dataset
from src.cleaner import clean_text
from tqdm import tqdm
import json

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=None, help="Number of entries to process")
parser.add_argument("--ids", nargs="+", type=int, help="Specific entry IDs to process")
args = parser.parse_args()

# Create output directory
os.makedirs("data", exist_ok=True)

# Load full ILC dataset
print("📦 Loading ILC dataset...")
dataset = load_dataset("d0r1h/ILC")
train_split = dataset["train"]

# Determine entries to process
all_indices = list(range(len(train_split)))
if args.ids:
    indices_to_process = [i for i in args.ids if i < len(train_split)]
elif args.n:
    indices_to_process = all_indices[:args.n]
else:
    indices_to_process = all_indices

print(f"📊 Total records to process: {len(indices_to_process)}")

# Clean entries
cleaned = []

for idx in tqdm(indices_to_process, desc="Cleaning ILC dataset"):
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

# Save JSON output
output_file = "data/cleaned_ilc.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(cleaned, f, indent=2, ensure_ascii=False)

print(f"✅ Cleaned ILC dataset saved to → {output_file}")
