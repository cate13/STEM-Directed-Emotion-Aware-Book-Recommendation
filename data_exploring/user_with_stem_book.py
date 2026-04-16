import json
import os
from collections import Counter

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CURATED_USERS_PATH = os.path.join(
    BASE_DIR, "processed_data", "curated_users.jsonl"
)

STEM_BOOKS_PATH_1 = os.path.join(
    BASE_DIR, "processed_data", "stem_isbns_from_classifier.txt"
)

STEM_BOOKS_PATH_2 = os.path.join(
    BASE_DIR, "processed_data", "stem_isbns_from_cosine.txt"
)

STEM_BOOKS_PATH_3 = os.path.join(
    BASE_DIR, "processed_data", "stem_isbns_from_topic.txt"
)

def load_stem_isbns(stem_paths):
    """Load all STEM ISBNs into a set for fast lookup."""
    stem_isbns = set()

    for path in stem_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                stem_isbns.add(line.strip())

    return stem_isbns


def compute_stem_counts_with_ratings(users_path, stem_isbns):
    """
    For each user, count how many STEM books they have read.
    Returns a Counter mapping:
        number_of_stem_books -> number_of_users
    """
    distribution = Counter()

    with open(users_path, "r", encoding="utf-8") as f:
        for line in f:
            user = json.loads(line)

            stem_count = 0
            for book in user["book_list"]:
                if book["isbn"] in stem_isbns and book["rating"] > 0:
                    stem_count += 1

            distribution[stem_count] += 1

    return distribution

def compute_stem_counts(users_path, stem_isbns):
    """
    For each user, count how many STEM books they have read.
    Returns a Counter mapping:
        number_of_stem_books -> number_of_users
    """
    distribution = Counter()

    with open(users_path, "r", encoding="utf-8") as f:
        for line in f:
            user = json.loads(line)

            stem_count = 0
            for book in user["book_list"]:
                if book["isbn"] in stem_isbns:
                    stem_count += 1

            distribution[stem_count] += 1

    return distribution


def get_youth_users_with_STEM():
    stem_isbns = load_stem_isbns([STEM_BOOKS_PATH_1, STEM_BOOKS_PATH_2, STEM_BOOKS_PATH_3])
    
    out_file_path = os.path.join(
        BASE_DIR, "data_exploring", "users_stem_books_with_rating.txt"
    )

    output = []

    with open(CURATED_USERS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            user = json.loads(line)
            user_id = user["user_id"]
            for book in user["book_list"]:
                if book["isbn"] in stem_isbns and book["rating"] > 0:
                    output.append(f"user {user_id}, isbn: {book['isbn']}, rating: {book['rating']}")

    with open(out_file_path, 'w') as f:
        for line in output:
            f.write(str(line) + "\n")


def get_youth_users_with_STEM():
    stem_isbns = load_stem_isbns([STEM_BOOKS_PATH_1, STEM_BOOKS_PATH_2, STEM_BOOKS_PATH_3])

    out_file_path = os.path.join(
        BASE_DIR, "data_exploring", "users_with_one_STEM_book.txt"
    )

    users = set()

    with open(CURATED_USERS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            user = json.loads(line)

            stem_count = 0
            for book in user["book_list"]:
                if book["isbn"] in stem_isbns and book["rating"] > 0:
                    stem_count += 1

            if stem_count == 1:
                users.add(user["user_id"])
    
    with open(out_file_path, 'w') as f:
        for line in users:
            f.write(str(line) + "\n")



def save_youth_stem_books():
    stem_isbns = load_stem_isbns([STEM_BOOKS_PATH_1, STEM_BOOKS_PATH_2, STEM_BOOKS_PATH_3])
    out_file_path = "stem_books_read_by_youth.txt"

    youth_stem_isbns = set()

    with open(CURATED_USERS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            user = json.loads(line)
            for book in user["book_list"]:
                if book["isbn"] in stem_isbns and book["rating"] > 0:
                    youth_stem_isbns.add(book["isbn"])


    with open(out_file_path, 'w') as f:
        for isbn in youth_stem_isbns:
            f.write(str(isbn) + "\n")

def main():
    stem_isbns = load_stem_isbns([STEM_BOOKS_PATH_1, STEM_BOOKS_PATH_2, STEM_BOOKS_PATH_3])

    distribution = compute_stem_counts_with_ratings(CURATED_USERS_PATH, stem_isbns)

    print("STEM Book Distribution (Number of STEM books → Number of Users)\n")
    for stem_count in sorted(distribution):
        print(f"{stem_count} → {distribution[stem_count]}")


if __name__ == "__main__":
    main()