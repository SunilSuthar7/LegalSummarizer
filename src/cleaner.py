import re

def clean_text(text: str) -> str:
    # 1. Remove excessive newlines
    text = re.sub(r'\n+', ' ', text)

    # 2. Normalize special characters
    text = re.sub(r'[“”]', '"', text)
    text = re.sub(r"[’‘]", "'", text)
    text = re.sub(r"[‐–—]", "-", text)

    # 3. Remove specific legal header phrases (only if they appear alone)
    text = re.sub(r"(?i)^\s*(CIVIL|CRIMINAL|APPEAL|WRIT|PETITION).*?\.\s*", '', text)
    text = re.sub(r"(?i)\bIN THE HIGH COURT OF [A-Z\s]+\.?", '', text)

    # 4. Collapse multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)

    return text.strip()
