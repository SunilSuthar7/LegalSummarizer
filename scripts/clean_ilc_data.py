import pandas as pd
import re
import os
from tqdm import tqdm

# Step 1: Load the raw dataset
print("📂 Loading raw_ilc.csv...")
df = pd.read_csv("../data/raw_ilc.csv")

# Step 2: Define cleaning functions
def fix_encoding(text):
    try:
        return text.encode("latin1", "ignore").decode("utf-8", "ignore")
    except:
        return text  # in case it's already clean

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

# Step 3: Apply cleaning to both columns
print("🧹 Cleaning input_text and summary_text...")
tqdm.pandas()

# If column names are different in your CSV (like Title, Summary Case), adjust these:
df['input_text'] = df.iloc[:, 0].progress_apply(clean_text)
df['summary_text'] = df.iloc[:, 1].progress_apply(clean_text)

# Optional: keep only the cleaned columns
df = df[['input_text', 'summary_text']]

# Step 4: Save the cleaned dataset
os.makedirs("../outputs", exist_ok=True)
output_path = "../outputs/cleaned_ILC_final.csv"
df.to_csv(output_path, index=False)
print(f"✅ Cleaned dataset saved to: {output_path}")

# Step 5: Print 5 random cleaned samples
print("\n🔍 Sample Cleaned Entries:")
print(df.sample(5))
