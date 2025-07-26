from datasets import load_dataset
import pandas as pd
import os

# Step 1: Load dataset from Hugging Face
print("📦 Loading ILC dataset...")
dataset = load_dataset("d0r1h/ILC")
train_data = dataset["train"]

# Step 2: Convert to pandas DataFrame
print("📊 Converting to pandas DataFrame...")
df = pd.DataFrame(train_data)

# Step 3: Save to CSV
os.makedirs("../data", exist_ok=True)
output_path = "../data/raw_ilc.csv"
df.to_csv(output_path, index=False)

print(f"✅ Done! Saved cleaned dataset to: {output_path}")
