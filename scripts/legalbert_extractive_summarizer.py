import json
import os
from tqdm import tqdm
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# --- Config ---
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
DATASETS = {
    "inabs": "data/legalbert_ready/inabs_legalbert_ready.json",
    "ilc": "data/legalbert_ready/ilc_legalbert_ready.json"
}
OUTPUT_DIR = "data/summaries_legalbert"
TOP_K = 3
MAX_DOCS = 100  # Number of samples to test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

def embed_sentence(sentence):
    tokens = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
    return cls_embedding.squeeze().cpu().numpy()

def summarize_text(text, top_k=TOP_K):
    sentences = sent_tokenize(text)
    if len(sentences) <= top_k:
        return text  # Not enough to summarize

    embeddings = [embed_sentence(s) for s in sentences]
    doc_embedding = np.mean(embeddings, axis=0).reshape(1, -1)
    sentence_embeddings = np.vstack(embeddings)

    sims = cosine_similarity(sentence_embeddings, doc_embedding).flatten()
    top_indices = sims.argsort()[-top_k:][::-1]
    top_sentences = [sentences[i] for i in sorted(top_indices)]

    return " ".join(top_sentences)

def summarize_dataset(name, path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for doc in tqdm(data[:MAX_DOCS], desc=f"Summarizing {name.upper()}", dynamic_ncols=True, leave=False):
        doc_id = doc.get("id")
        text = doc.get("input_text", "").strip()
        if len(text.split()) < 50:
            results.append({
                "id": doc_id,
                "status": "skipped",
                "reason": "too short"
            })
            continue

        try:
            summary = summarize_text(text)
            results.append({
                "id": doc_id,
                "status": "success",
                "summary": summary,
                "original_length": len(text.split()),
                "summary_length": len(summary.split())
            })
        except Exception as e:
            results.append({
                "id": doc_id,
                "status": "error",
                "error": str(e),
                "sample_text": text[:200]
            })

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{name}_legalbert_summaries.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Finished {name.upper()} → {out_path}")

def main():
    for name, path in DATASETS.items():
        summarize_dataset(name, path)

if __name__ == "__main__":
    main()
