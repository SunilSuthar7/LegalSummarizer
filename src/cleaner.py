import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data (only once)
nltk.download("punkt")
nltk.download("stopwords")

def normalize_quotes(text):
    return text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

def remove_case_numbers(text):
    return re.sub(r"case\s*(no\.|number)?[\s:]*\d+", "", text, flags=re.IGNORECASE)

def remove_legal_headers(text):
    text = re.sub(r"(?i)^\s*(CIVIL|CRIMINAL|APPEAL|WRIT|PETITION).*?\.\s*", '', text)
    text = re.sub(r"(?i)\bIN THE HIGH COURT OF [A-Z\s]+\.?", '', text)
    return text

def remove_special_characters(text):
    return re.sub(r"[^\w\s.,;:]", "", text)

def standardize_spacing(text):
    return re.sub(r"\s{2,}", " ", text)

def remove_stopwords(text, lang="english"):
    words = word_tokenize(text)
    return " ".join(w for w in words if w.lower() not in stopwords.words(lang))

def clean_text(text: str, aggressive: bool = False) -> str:
    if not isinstance(text, str):
        return ""

    # Base cleaning (always applied)
    text = text.replace('\n', ' ')
    text = normalize_quotes(text)
    text = remove_legal_headers(text)
    text = remove_case_numbers(text)
    text = standardize_spacing(text)

    if aggressive:
        text = text.lower()
        text = remove_special_characters(text)
        text = remove_stopwords(text)

    return text.strip()
