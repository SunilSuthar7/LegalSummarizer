# src/tokenizer.py

import nltk
import string

# Ensure tokenizer models are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def word_tokenize_nltk(text):
    """
    Tokenizes input text using NLTK's word_tokenize.
    """
    from nltk.tokenize import word_tokenize
    if not isinstance(text, str) or not text.strip():
        return []
    return word_tokenize(text)


def remove_punctuation(tokens):
    """
    Removes punctuation tokens from a list of tokens.
    """
    if not isinstance(tokens, list):
        return []

    return [token for token in tokens if token not in string.punctuation]
