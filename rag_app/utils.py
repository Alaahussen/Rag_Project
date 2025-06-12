import re
import spacy
from spacy.cli import download
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def split_camel_case(text):
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

def extract_name_spacy(text):
    text = re.sub(r'\.pdf$', '', text, flags=re.IGNORECASE)
    text = split_camel_case(text)
    text = re.sub(r'[_&\-]', ' ', text)
    text = re.sub(r'\b(cv|resume|file|document|data|engineer|developer|ai|ml|scientist|al)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()  # clean up spacing

    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.strip()

    # fallback: try title-casing alpha words
    words = [w.capitalize() for w in text.split() if w.isalpha()]
    return " ".join(words) if words else None


def extract_matched_names_from_query(query, full_name_dict, first_name_dict):
    query = query.lower()
    matched = [name for name in full_name_dict if name in query]
    if not matched:
        matched = [name for name in first_name_dict if name in query]
    return matched
