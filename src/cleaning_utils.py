import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Automatically download required NLTK resources
nltk.download("stopwords")
nltk.download("punkt")

def normalize_text(text):
    return text.lower().strip()

def remove_special_characters(text):
    return re.sub(r"[^\w\s.,;:]", "", text)

def remove_stopwords(text, lang="english"):
    words = word_tokenize(text, preserve_line=True)
    filtered = [w for w in words if w.lower() not in stopwords.words(lang)]
    return " ".join(filtered)


def standardize_spacing(text):
    return re.sub(r"\s+", " ", text)

def remove_case_numbers(text):
    return re.sub(r"case\s*(no\.|number)?[\s:]*\d+", "", text, flags=re.IGNORECASE)

def remove_headers_footers(text):
    patterns = [
        r"in the [a-z\s]* court of [a-z\s]*",  # court headers
        r"judgment delivered on\.",           # footers
    ]
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    return text

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = normalize_text(text)
    text = remove_headers_footers(text)
    text = remove_case_numbers(text)
    text = remove_special_characters(text)
    text = standardize_spacing(text)
    text = remove_stopwords(text)
    return text.strip()

def clean_dataframe(df, column="input_text"):
    df["cleaned_text"] = df[column].apply(clean_text)
    return df
