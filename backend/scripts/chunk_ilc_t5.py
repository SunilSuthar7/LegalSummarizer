import json
import argparse
from transformers import T5Tokenizer
from tqdm import tqdm

# CONFIG
INPUT_PATH = "data/cleaned_ilc.json"
OUTPUT_PATH = "data/chunked_ilc.json"
TEXT_KEY = "input_text"
MODEL_NAME = "t5-base"
MAX_TOKENS = 512

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

def chunk_text_t5(text, max_tokens=MAX_TOKENS):
    words = text.split()
    chunks = []
    current_chunk = []
    current_len = 0

    for word in words:
        token_len = len(tokenizer.encode(word, add_special_tokens=False))
        if current_len + token_len > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_len = token_len
        else:
            current_chunk.append(word)
            current_len += token_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=None)
parser.add_argument("--ids", nargs="+", type=int)
args = parser.parse_args()

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Select subset
if args.ids:
    data = [d for d in data if d["id"] in args.ids]
elif args.n:
    data = data[:args.n]

output_data = []
for entry in tqdm(data, desc="Chunking entries"):
    eid = entry.get("id")
    text = entry.get(TEXT_KEY, "").strip()
    if not text:
        continue
    chunks = chunk_text_t5(text, max_tokens=MAX_TOKENS)
    output_data.append({"id": eid, "chunks": chunks})

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"✅ Chunked {len(output_data)} entries and saved to {OUTPUT_PATH}")
