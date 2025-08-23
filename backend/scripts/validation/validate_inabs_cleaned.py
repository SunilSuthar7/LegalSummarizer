import pandas as pd
import os

# === Path to your cleaned IN-ABS dataset ===
INABS_PATH = "data/sample_cleaned_inabs.json"

# === Load and Validate ===
def validate_inabs(path):
    print(f"\n🔍 Validating cleaned IN-ABS dataset at: {path}")

    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return

    try:
        df = pd.read_json(path)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return

    # Check required columns
    required_cols = ['id', 'input_text', 'summary_text']
    if list(df.columns) != required_cols:
        print("❌ Column mismatch:")
        print("Expected:", required_cols)
        print("Found:   ", list(df.columns))
        return

    # Check for nulls
    if df[['input_text', 'summary_text']].isnull().any().any():
        print("❌ Null values found in input_text or summary_text.")
        return

    # Print text length stats
    print("✅ Schema OK. Calculating statistics...")
    print("\n📏 input_text lengths:")
    print(df['input_text'].str.len().describe())
    print("\n📏 summary_text lengths:")
    print(df['summary_text'].str.len().describe())

    # Spot check sample rows
    print("\n🧾 Spot Check of Samples:")
    for i in [0, 3, 7]:
        if i < len(df):
            print(f"\n🔹 ID {df.loc[i, 'id']}")
            print("Input Text:", df.loc[i, 'input_text'][:250], "...")
            print("Summary:", df.loc[i, 'summary_text'][:150])

# Run the validation
validate_inabs(INABS_PATH)
