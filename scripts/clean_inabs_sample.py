import sys
import os

# Add root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets import load_dataset
from src.cleaner import clean_text
import pandas as pd
from tqdm import tqdm
import os

# Create output directory
os.makedirs("data", exist_ok=True)

# Load dataset
print("Loading IN-ABS dataset...")
dataset = load_dataset("percins/IN-ABS")
train_split = dataset["train"]

# We'll process only the first 100 samples
sample_cleaned = []

print("Cleaning samples...")
for idx in tqdm(range(100), desc="Cleaning sample"):
    raw = train_split[idx]
    cleaned = {
        "id": idx,
        "input_text": clean_text(raw["text"]),               # Safe readable mode
        "summary_text": clean_text(raw["summary"])
    }
    sample_cleaned.append(cleaned)

# Save to JSON only
output_path = "data/sample_cleaned_inabs.json"
df = pd.DataFrame(sample_cleaned)
df.to_json(output_path, orient="records", indent=2, force_ascii=False)

print(f" Cleaned data saved to → {output_path}")
