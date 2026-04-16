import json
import math
import os
import random
from collections import Counter

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STEM_BOOKS_PATH = os.path.join(
    BASE_DIR, "processed_data", "stem_books.jsonl"
)

def load_stem_isbns(stem_path):
    """Load all STEM ISBNs into a set for fast lookup."""
    stem_isbns = set()

    with open(stem_path, "r", encoding="utf-8") as f:
        for line in f:
            book = json.loads(line)
            stem_isbns.add(book["ISBN"])

    return stem_isbns

MISSING_BOOKS_PATH = os.path.join(
    BASE_DIR, "data_exploring", "missing_isbns.txt"
)

def load_missing_isbns():
    """Load all STEM ISBNs into a set for fast lookup."""
    missing_isbns = set()

    with open(MISSING_BOOKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            missing_isbns.add(line.strip())

    return missing_isbns

jsonl_input = os.path.join(
    BASE_DIR, "processed_data", "curated_users.jsonl"
)

jsonl_output = os.path.join(
    BASE_DIR, "user_eval_sets", "users_with_two_three_STEM_books.jsonl"
)

stem_isbns = load_stem_isbns(STEM_BOOKS_PATH)
missing_description = load_missing_isbns()


with open(jsonl_input, "r") as infile, open(jsonl_output, "w") as outfile:
    for line in infile:
        user = json.loads(line)

        cleaned_books = [
            book for book in user.get("book_list", [])
            if book["isbn"] not in missing_description
        ]

        matching_books = [
            book for book in cleaned_books
            if book["isbn"] in stem_isbns and book.get("rating", 0) > 0
        ]

        if 2 <= len(matching_books) <= 3:
            user["book_list"] = cleaned_books 
            outfile.write(json.dumps(user) + "\n")