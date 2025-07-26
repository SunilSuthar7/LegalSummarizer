from datasets import load_dataset
from src.cleaner import clean_text
import pandas as pd
from tqdm import tqdm
import os

# Create output directory
os.makedirs("data", exist_ok=True)

# Load dataset
dataset = load_dataset("percins/IN-ABS")
train_split = dataset["train"]

# We'll process only the first 100 samples
sample_cleaned = []

for idx in tqdm(range(100), desc="Cleaning sample"):
    raw = train_split[idx]
    cleaned = {
        "id": idx,
        "input_text": clean_text(raw["text"]),
        "summary_text": clean_text(raw["summary"])
    }
    sample_cleaned.append(cleaned)

# Save to JSON and CSV
df = pd.DataFrame(sample_cleaned)
df.to_json("data/sample_cleaned_inabs.json", orient="records", indent=2, force_ascii=False)
df.to_csv("data/sample_cleaned_inabs.csv", index=False)

print("✅ Sample cleaned data saved to data/sample_cleaned_inabs.json and .csv")
