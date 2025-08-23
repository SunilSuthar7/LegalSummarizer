import json

# Load the cleaned dataset
with open("data/sample_cleaned_ilc.json", encoding="utf-8") as f:
    data = json.load(f)

# Change this ID to see any entry
i = 5  # 👈 You can change this to 0, 1, 200, etc.

# Safety check
if i >= len(data):
    print(f"❌ Entry ID {i} not found. Dataset only has {len(data)} entries.")
else:
    print(f"\n🆔 ID: {i}")
    print(f"\n📄 Full input_text:\n{data[i]['input_text']}\n")
    print(f"📝 Summary:\n{data[i]['summary_text']}\n")
