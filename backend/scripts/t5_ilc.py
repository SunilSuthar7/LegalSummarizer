import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
import re
import time
import argparse

INPUT_PATH = "data/chunked_ilc.json"
OUTPUT_PATH = "data/t5_ilc_final.json"
MODEL_NAME = "t5-base"

MAX_INPUT_TOKENS = 512
CHUNK_SUM_MAX = 100
FINAL_SUM_MAX = 300
FINAL_MIN_LEN = 90
NUM_BEAMS = 8
LENGTH_PENALTY = 1.0
BATCH_SIZE = 4
SLEEP_BETWEEN_BATCHES = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
model.eval()

def join_chunks(chunks):
    return " ".join(chunks)

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=None)
parser.add_argument("--ids", nargs="+", type=int)
args = parser.parse_args()

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# subset selection
if args.ids:
    data = [d for d in data if d["id"] in args.ids]
elif args.n:
    data = data[:args.n]

results = []
for entry in tqdm(data, desc="Summarizing entries"):
    eid = entry.get("id")
    chunks = entry.get("chunks", [])
    full_text = join_chunks(chunks) if chunks else ""
    # Simple single-stage summary for demo; your existing two-stage logic can remain
    input_enc = tokenizer.encode("summarize: " + full_text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS).to(device)
    summary_ids = model.generate(input_enc, max_length=FINAL_SUM_MAX, min_length=FINAL_MIN_LEN, num_beams=NUM_BEAMS)
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    results.append({"id": eid, "refined_summary_improved": summary_text})

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✅ Saved {len(results)} summaries to {OUTPUT_PATH}")
