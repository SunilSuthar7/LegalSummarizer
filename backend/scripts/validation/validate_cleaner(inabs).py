import os
import re
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

# -------------------------------
# Load Raw and Cleaned Datasets
# -------------------------------
# Load raw dataset from Hugging Face cache (IN-ABS original)
raw_ds = load_dataset("percins/IN-ABS", split="train")
raw_df = pd.DataFrame(raw_ds)
# Rename raw columns to match cleaned_df structure
raw_df = raw_df.rename(columns={"text": "input_text", "summary": "summary_text"})



# Load cleaned dataset from your local file
cleaned_path = os.path.join("data", "cleaned_inabs.json")
cleaned_df = pd.read_json(cleaned_path)

# Align datasets
min_len = min(len(raw_df), len(cleaned_df))
raw_df = raw_df.iloc[:min_len].reset_index(drop=True)
cleaned_df = cleaned_df.iloc[:min_len].reset_index(drop=True)

# -------------------------------
# Legal Keywords & Noise Patterns
# -------------------------------
LEGAL_TERMS = {
    'section', 'article', 'act', 'clause',
    'vs', 'versus', 'schedule', 'appellant', 'respondent'
}

def noise_present(text: str) -> bool:
    return bool(re.search(r"case\s*no|in the\s+(high\s+)?court", text or "", re.I))

# -------------------------------
# Validation Loop
# -------------------------------
results = []
for i in tqdm(range(min_len), desc="Validating"):
    raw_input = raw_df.loc[i, "input_text"]
    raw_summary = raw_df.loc[i, "summary_text"]
    cleaned_input = cleaned_df.loc[i, "input_text"]
    cleaned_summary = cleaned_df.loc[i, "summary_text"]

    results.append({
        "input_raw_len": len(raw_input or ""),
        "input_cleaned_len": len(cleaned_input or ""),
        "summary_raw_len": len(raw_summary or ""),
        "summary_cleaned_len": len(cleaned_summary or ""),
        "input_preserved_legal": any(term in (cleaned_input or "").lower() for term in LEGAL_TERMS),
        "summary_preserved_legal": any(term in (cleaned_summary or "").lower() for term in LEGAL_TERMS),
        "input_noise_removed": not noise_present(cleaned_input),
        "summary_noise_removed": not noise_present(cleaned_summary),
    })

result_df = pd.DataFrame(results)

# -------------------------------
# Summary Statistics
# -------------------------------
print("\n=== Length Reduction ===")
print(result_df[[
    "input_raw_len", "input_cleaned_len",
    "summary_raw_len", "summary_cleaned_len"
]].describe())

print("\n=== Legal Term Preservation Rate ===")
print("Input Text   :", f"{result_df['input_preserved_legal'].mean():.2%}")
print("Summary Text :", f"{result_df['summary_preserved_legal'].mean():.2%}")

print("\n=== Noise Removal Rate ===")
print("Input Text   :", f"{result_df['input_noise_removed'].mean():.2%}")
print("Summary Text :", f"{result_df['summary_noise_removed'].mean():.2%}")