import pandas as pd
import os

# Set the path to the cleaned ILC file
file_path = "../data/sample_cleaned_ilc.csv"

# Check if file exists
if not os.path.exists(file_path):
    print(f"❌ File not found: {file_path}")
    exit()

# Load the cleaned CSV
df = pd.read_csv(file_path)

# Basic file info
print("📊 Loaded cleaned ILC dataset:")
print(f"→ Total Rows: {len(df)}")
print(f"→ Columns: {df.columns.tolist()}")

# Print 5 random rows
print("\n🔍 Sample Entries:")
print(df.sample(5))

# Optional: Show token length stats
df['input_len'] = df['input_text'].apply(lambda x: len(str(x).split()))
df['summary_len'] = df['summary_text'].apply(lambda x: len(str(x).split()))

print("\n📈 Token Length Stats:")
print(df[['input_len', 'summary_len']].describe())
