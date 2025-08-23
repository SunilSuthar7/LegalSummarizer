# scripts/extractive_summarizer.py

import json
import os
from tqdm import tqdm
import networkx as nx
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')

def load_tokenized_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_summary(summary_dict, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(summary_dict, f, indent=2)

def build_similarity_matrix(sentences):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix

def textrank_summarize(sentences, top_n=3):
    if len(sentences) <= top_n:
        return sentences

    sim_matrix = build_similarity_matrix(sentences)
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary = [sent for score, sent in ranked_sentences[:top_n]]
    return summary

def summarize_documents(input_path, output_path, top_n=3):
    data = load_tokenized_data(input_path)
    summary_data = {}

    for entry in tqdm(data, total=len(data), desc=f"Summarizing {os.path.basename(input_path)}"):
        doc_id = str(entry.get("id", len(summary_data)))
        tokens = entry.get("tokens", [])
        text = " ".join(tokens)
        sentences = sent_tokenize(text)

        if not sentences:
            summary_data[doc_id] = []
            continue

        summary = textrank_summarize(sentences, top_n=top_n)
        summary_data[doc_id] = summary

    save_summary(summary_data, output_path)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    summarize_documents("data/tokenized_inabs.json", "data/extractive_summary_inabs.json", top_n=3)
    summarize_documents("data/tokenized_ilc.json", "data/extractive_summary_ilc.json", top_n=3)
