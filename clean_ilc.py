from src.cleaning_utils import clean_dataframe
import pandas as pd
import os

# File paths
input_path = "data/cleaned_ILC_final.csv"
output_path = "cleaned/cleaned_ilc_output.csv"

# Make sure output folder exists
os.makedirs("cleaned", exist_ok=True)

# Read CSV
df = pd.read_csv(input_path)

# Apply cleaning
df_cleaned = clean_dataframe(df, column="input_text")  # Make sure the column is named 'input_text'

# Save cleaned file
df_cleaned.to_csv(output_path, index=False)

print(f" Cleaned ILC file saved to: {output_path}")
print(df_cleaned[["input_text", "cleaned_text"]].head())
