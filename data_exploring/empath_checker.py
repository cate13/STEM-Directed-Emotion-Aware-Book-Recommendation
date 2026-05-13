import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEM_TAG_PATH = os.path.join(BASE_DIR, "processed_data", "stem_isbns_from_topic.txt")
STEM_CLASSIFIER_PATH = os.path.join(BASE_DIR, "processed_data", "stem_isbns_from_classifier.txt")
STEM_COSINE_PATH = os.path.join(BASE_DIR, "processed_data", "stem_isbns_from_cosine.txt")
BOOK_VECTOR = os.path.join(BASE_DIR, "processed_data", "book_vectors_base.jsonl")
ONLY_STEM_ISBNS = os.path.join(BASE_DIR, "data_exploring", "stem_isbns_with_empath.jsonl")

def load_stem_isbns(stem_paths):
    """Load all STEM ISBNs into a set for fast lookup."""
    stem_isbns = set()

    for path in stem_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                stem_isbns.add(line.strip())

    return stem_isbns

stem_isbns = load_stem_isbns([STEM_TAG_PATH, STEM_CLASSIFIER_PATH, STEM_COSINE_PATH])

with open(BOOK_VECTOR, 'r') as source_file, open(ONLY_STEM_ISBNS, 'w') as dest_file:
    for line in source_file:
        book = json.loads(line)
        isbn = book["isbn"]
        if isbn in stem_isbns:
            empath = book["empath"]
            entry = {
                "isbn" : isbn,
                "empath" : empath,
            }
            json_line = json.dumps(entry)
            dest_file.write(json_line + "\n")