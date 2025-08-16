import json
from rouge_score import rouge_scorer

# ===== FILE PATHS =====
extractive_path = 'data/t5_ilc_final.json'  # Candidate/refined summaries
reference_path = 'data/cleaned_ilc.json'  # Ground truth summaries

# ===== LOAD FILES =====
with open(extractive_path, 'r', encoding='utf-8') as f:
    extractive_data = json.load(f)

with open(reference_path, 'r', encoding='utf-8') as f:
    reference_data = json.load(f)

# ===== GET ID LISTS =====
ref_ids = {entry['id'] for entry in reference_data if 'summary_text' in entry}
cand_ids = {entry['id'] for entry in extractive_data if 'refined_summary_improved' in entry}

print(f"\nReference IDs: {sorted(ref_ids)}")
print(f"Candidate IDs: {sorted(cand_ids)}")

matched_ids = ref_ids & cand_ids
missing_in_cand = ref_ids - cand_ids
missing_in_ref = cand_ids - ref_ids

print(f"\nMatched IDs: {sorted(matched_ids)}")
print(f"Missing in candidate file: {sorted(missing_in_cand)}")
print(f"Missing in reference file: {sorted(missing_in_ref)}\n")

# ===== BUILD REFERENCE DICT =====
reference_dict = {entry['id']: entry['summary_text'] for entry in reference_data if 'summary_text' in entry}

# ===== ROUGE SETUP =====
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

processed = skipped_no_candidate = skipped_no_ref = 0

for entry in extractive_data:
    eid = entry['id']
    extractive_summary = entry.get('refined_summary_improved', '')
    reference_summary = reference_dict.get(eid, '')

    if not extractive_summary:
        skipped_no_candidate += 1
        continue
    if not reference_summary:
        skipped_no_ref += 1
        continue

    score = scorer.score(reference_summary, extractive_summary)
    for key in scores:
        scores[key].append(score[key].fmeasure)
    processed += 1

# ===== RESULTS =====
print(f"Processed entries scored: {processed}")
print(f"Skipped (no candidate field or empty): {skipped_no_candidate}")
print(f"Skipped (no matching reference): {skipped_no_ref}")

print("\nROUGE scores (F1):")
for key in scores:
    avg = sum(scores[key]) / len(scores[key]) if scores[key] else 0
    print(f"{key}: {avg*100:.2f}%")  # convert to %