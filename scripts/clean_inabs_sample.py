import sys
import os

# Add root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import load_dataset
from src.cleaner import clean_text
import pandas as pd
from tqdm import tqdm

# Create output directory if not exists
os.makedirs("data", exist_ok=True)

# Load full IN-ABS dataset
print(" Loading full IN-ABS dataset...")
dataset = load_dataset("percins/IN-ABS")
train_split = dataset["train"]

# Clean all samples
cleaned_all = []
print("Cleaning full dataset...")
for idx in tqdm(range(len(train_split)), desc="Cleaning"):
    raw = train_split[idx]
    cleaned = {
        "id": idx,
        "input_text": clean_text(raw["text"]),
        "summary_text": clean_text(raw["summary"])
    }
    cleaned_all.append(cleaned)

# Save to JSON
output_path = "data/cleaned_inabs.json"
df = pd.DataFrame(cleaned_all)
df.to_json(output_path, orient="records", indent=2, force_ascii=False)

print(f"\nFull cleaned IN-ABS data saved to → {output_path}")
print(f"Total entries cleaned: {len(cleaned_all)}")