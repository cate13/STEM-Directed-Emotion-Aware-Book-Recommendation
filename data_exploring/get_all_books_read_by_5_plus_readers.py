import json
import os
from collections import Counter
from tqdm import tqdm

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CURATED_USERS_PATH = os.path.join(
    BASE_DIR, "processed_data", "curated_users_rated_stem_books.jsonl"
)

all_isbn = set()

with open(CURATED_USERS_PATH, "r", encoding="utf-8") as f:
    for line in tqdm(f):
        user = json.loads(line)
        for book in user["book_list"]:
            isbn = book["isbn"]
            all_isbn.add(isbn)

file_path = "books_read_by_readers_who_read_stem.txt"

with open(file_path, 'w') as f:
    for isbn in tqdm(all_isbn):
        f.write(str(isbn) + "\n")
