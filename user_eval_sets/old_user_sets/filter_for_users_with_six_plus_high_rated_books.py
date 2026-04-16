import json
import os
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

JSONL_PATH = os.path.join(
    BASE_DIR, "user_eval_sets", "users_with_two_three_STEM_books.jsonl"
)

six_plus_high_rated_books = []

missing_isbns = load_missing_isbns()

with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        user = json.loads(line)

        count = 0
        for book in user["book_list"]:
            if book['isbn'] in missing_isbns:
                continue
            if book["rating"] >= 7:
                count += 1

        if count >= 6:
            six_plus_high_rated_books.append(line)

out_file_name = "user_eval_sets/users_with_two_three_STEM_books_and_six_plus_high_rated_books.jsonl"

num_users = 0
with open(out_file_name, 'w') as f:
    for line in six_plus_high_rated_books:
        num_users += 1
        data = json.loads(line)
        data["book_list"] = [
            book for book in data["book_list"]
            if book.get("rating", 0) != 0
            and book.get("isbn") not in missing_isbns
        ]
        f.write(json.dumps(data) + "\n")

print(f"{num_users} Users")