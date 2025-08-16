import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List
import re
from tqdm import tqdm
import time

# ===== CONFIG =====
INPUT_PATH = "data/chunked_ilc.json"
OUTPUT_PATH = "data/t5_ilc_final.json"
MODEL_NAME = "t5-base"

MAX_INPUT_TOKENS = 512       # T5 input limit
CHUNK_SUM_MAX = 100
FINAL_SUM_MAX = 300
FINAL_MIN_LEN = 90
NUM_BEAMS = 8
LENGTH_PENALTY = 1.0
KEYWORD_SENT_LIMIT = 5
TEST_COUNT = 100           # None = full dataset
BATCH_SIZE = 4
SLEEP_BETWEEN_BATCHES = 1    # seconds; optional sleep
# ==================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
model.eval()

KEYWORDS = [
    'mediation', 'conciliation', 'FIR', 'settlement', 'agreed',
    'section', 'sections', '498A', '323', '354', '504', 'arbitration',
    'settlement agreement', 'inherent power', 'Full Bench', 'Ram Lal',
    'tribunal', 'appeal', 'supreme court', 'judgment', 'petition'
]

def join_chunks(chunks: List[str]) -> str:
    return ' '.join(chunks)

def split_into_sentences(text: str) -> List[str]:
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if s.strip()]

def find_keyword_sentences(text: str, limit: int) -> List[str]:
    sents = split_into_sentences(text)
    found = []
    lowered = [s.lower() for s in sents]
    for kw in KEYWORDS:
        for i, s in enumerate(lowered):
            if kw in s and sents[i] not in found:
                found.append(sents[i])
                if len(found) >= limit:
                    return found
    return found

def summarize_text_batch(batch_texts: List[str], max_length: int, min_length: int = 10) -> List[str]:
    with torch.cuda.amp.autocast():  # FP16
        enc = tokenizer(["summarize: " + t for t in batch_texts],
                        return_tensors="pt",
                        max_length=MAX_INPUT_TOKENS,
                        truncation=True,
                        padding=True).to(device)
        out = model.generate(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            max_length=max_length,
            min_length=min_length,
            num_beams=NUM_BEAMS,
            length_penalty=LENGTH_PENALTY,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        summaries = [tokenizer.decode(o, skip_special_tokens=True) for o in out]
    torch.cuda.empty_cache()
    if SLEEP_BETWEEN_BATCHES > 0:
        time.sleep(SLEEP_BETWEEN_BATCHES)
    return summaries

def adaptive_group_chunks(chunks: List[str]) -> List[str]:
    groups = []
    current_group = []
    current_tokens = 0
    for chunk in chunks:
        chunk_tokens = len(tokenizer.encode(chunk))
        if current_tokens + chunk_tokens > MAX_INPUT_TOKENS:
            groups.append(' '.join(current_group))
            current_group = [chunk]
            current_tokens = chunk_tokens
        else:
            current_group.append(chunk)
            current_tokens += chunk_tokens
    if current_group:
        groups.append(' '.join(current_group))
    return groups

def two_stage_summarize(full_text: str, chunks: List[str]) -> str:
    grouped_chunks = adaptive_group_chunks(chunks) if chunks else [full_text]
    # Stage 1: batch summarize grouped chunks
    chunk_summaries = []
    for i in range(0, len(grouped_chunks), BATCH_SIZE):
        batch = grouped_chunks[i:i+BATCH_SIZE]
        batch_summaries = summarize_text_batch(batch, max_length=CHUNK_SUM_MAX, min_length=20)
        chunk_summaries.extend(batch_summaries)
    # Stage 2: combine and summarize again
    combined = ' '.join(chunk_summaries)
    final_summary = summarize_text_batch([combined], max_length=FINAL_SUM_MAX, min_length=FINAL_MIN_LEN)[0]
    # Keyword sentence preservation
    keyword_sents = find_keyword_sentences(full_text, limit=KEYWORD_SENT_LIMIT)
    prepend_sents = [ks for ks in keyword_sents if ks not in final_summary]
    if prepend_sents:
        final_summary = ' '.join(prepend_sents) + ' ' + final_summary
    return re.sub(r'\s+', ' ', final_summary).strip()

def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if TEST_COUNT:
        data = data[:TEST_COUNT]

    results = []
    for entry in tqdm(data, desc="Summarizing ILC", ncols=100):
        eid = entry.get("id")
        chunks = entry.get("chunks", [])
        full_text = join_chunks(chunks) if chunks else entry.get("summary", "")
        refined_summary = two_stage_summarize(full_text, chunks)
        results.append({"id": eid, "refined_summary_improved": refined_summary})

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved {len(results)} summaries to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()