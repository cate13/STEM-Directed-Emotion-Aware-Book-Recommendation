import json
import os
from collections import Counter

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_VECTOR = os.path.join(
    BASE_DIR, "processed_data", "book_vectors_base_missing.jsonl"
)

EXTENDED_VECTOR = os.path.join(
    BASE_DIR, "processed_data", "book_vectors_extended.jsonl"
)

BOOKS_READ_BY_TEENS = os.path.join(
    BASE_DIR, "processed_data", "curated_users.jsonl"
)

def extract_isbns_from_reader():
    isbns = set()

    with open(BOOKS_READ_BY_TEENS, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            book_list = data.get('book_list')
            for book in book_list:
                i = book.get('isbn')
                isbns.add(i)
    return isbns

def get_set(path, key):
    isbns = set()

    with open(path, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            i = data.get(key)
            isbns.add(i)
    return isbns


base_set = get_set(BASE_VECTOR, "isbn")
#extended_set = get_set(EXTENDED_VECTOR, "isbn")


# only_in_base = base_set - extended_set
# # 2. Items in extended_set but NOT in base_set
# only_in_extended = extended_set - base_set

book_teens_have_read = extract_isbns_from_reader()

book_teens_have_read_that_have_not_been_vectorized = book_teens_have_read - base_set

print(f"There are {len(book_teens_have_read_that_have_not_been_vectorized)} that have been read by teens but not vectorized")

with open("isbns_to_vectorize.txt", 'w') as f:
        for isbn in book_teens_have_read_that_have_not_been_vectorized:
            f.write(str(isbn) + "\n")

# print(f"There are {len(only_in_base)} isbns missing from extended")
# print(f"There are {len(only_in_extended)} isbns missing from base")

# print(f"Only in base:")       
# for i in only_in_base:
#     print(f"\t{i}")

# print(f"Only in extended:") 
# for i in only_in_extended:
#     print(f"\t{i}")
