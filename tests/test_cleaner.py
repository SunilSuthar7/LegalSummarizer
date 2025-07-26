import sys
import os

# ✅ Add src/ to Python import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# ✅ Now import clean_text from src/cleaner.py
from cleaner import clean_text

def test_basic_cleanup():
    input_text = """
    CIVIL APPEAL No. 1234 of 1990.
    IN THE HIGH COURT OF DELHI.

    The appellant filed the petition — citing FR 54.
    """
    cleaned = clean_text(input_text)

    assert "CIVIL APPEAL" not in cleaned
    assert "HIGH COURT" not in cleaned
    assert "appellant filed the petition" in cleaned
    assert "\n" not in cleaned
    assert "  " not in cleaned
