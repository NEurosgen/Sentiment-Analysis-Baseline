import re
import string
from typing import Optional

PUNCT_TABLE = str.maketrans('', '', string.punctuation)

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(PUNCT_TABLE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
