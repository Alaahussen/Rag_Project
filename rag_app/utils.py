import re
import spacy
import en_core_web_sm
try:
    nlp = en_core_web_sm.load()
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def split_camel_case(text):
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

def extract_name_spacy(text):
    text = re.sub(r'\.pdf$', '', text, flags=re.IGNORECASE)
    text = split_camel_case(text)
    text = re.sub(r'[_&\-]', ' ', text)
    text = re.sub(r'\b(cv|resume|file|document|data|engineer|developer|ai|ml|scientist)\b', '', text, flags=re.IGNORECASE)
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.strip()
    words = [w for w in text.split() if w.istitle()]
    return " ".join(words) if words else None

def extract_matched_names_from_query(query, full_name_dict, first_name_dict):
    query = query.lower()
    matched = [name for name in full_name_dict if name in query]
    if not matched:
        matched = [name for name in first_name_dict if name in query]
    return matched
