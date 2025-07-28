import json
import os

# Corrected relative file paths from /validation/ to /data/
inabs_path = os.path.join("..", "sample_cleaned_inabs.json")
ilc_path = os.path.join("..", "sample_cleaned_ilc.json")

def load_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"✅ Loaded {file_path} — {len(data)} records")
        return data
    except Exception as e:
        print(f"❌ Error loading {file_path}:", e)
        return None

def preview_sample(data, label):
    print(f"\n🔍 Sample Record from {label}:")
    if data and len(data) > 0:
        sample = data[0]
        for key, value in sample.items():
            print(f"  {key}: {str(value)[:100]}...")  # Truncate long text
    else:
        print(f"⚠️ No data found in {label}")

def check_field_consistency(data, label, expected_fields):
    print(f"\n🔎 Validating fields in {label} dataset...")

    total = len(data)
    errors = 0

    for idx, record in enumerate(data):
        # Check type
        if not isinstance(record, dict):
            print(f"❌ Record {idx} is not a dictionary")
            errors += 1
            continue

        # Check expected fields
        missing = [key for key in expected_fields if key not in record]
        if missing:
            print(f"❌ Record {idx} missing fields: {missing}")
            errors += 1

        # Check for empty or null values
        for key in expected_fields:
            if key in record and (record[key] is None or str(record[key]).strip() == ""):
                print(f"⚠️ Record {idx} has empty/null value for '{key}'")
                errors += 1

        # Optional: Check types (string)
        for key in ['input_text', 'summary_text']:
            if key in record and not isinstance(record[key], str):
                print(f"⚠️ Record {idx} field '{key}' is not a string")
                errors += 1

    print(f"✅ Checked {total} records — {errors} issues found in {label}")

if __name__ == "__main__":
    print("🔎 Loading Cleaned JSON Files for Validation\n")

    inabs_data = load_json(inabs_path)
    ilc_data = load_json(ilc_path)

    preview_sample(inabs_data, "IN-ABS")
    preview_sample(ilc_data, "ILC")

    required_fields = ["id", "input_text", "summary_text"]
    check_field_consistency(inabs_data, "IN-ABS", required_fields)
    check_field_consistency(ilc_data, "ILC", required_fields)
