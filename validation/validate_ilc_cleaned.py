import json

# Load the cleaned JSON
with open("../data/sample_cleaned_ilc.json", encoding="utf-8") as f:
    data = json.load(f)

# Change this ID to view a different entry
entry_id = 1  # 👈 You can change this to 0, 2, 3, etc.

# Safety check
if entry_id >= len(data):
    print(f"❌ Entry ID {entry_id} not found. Dataset only has {len(data)} entries.")
else:
    print(f"\n🧾 ID: {entry_id}")
    print("\n📄 Full input_text:\n")
    print(data[entry_id]['input_text'])
    print("\n📝 Summary:\n")
    print(data[entry_id]['summary_text'])
 u pt this code in my valiation file n check