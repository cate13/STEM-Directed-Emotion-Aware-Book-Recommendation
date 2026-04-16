import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CURATED_USERS_PATH = os.path.join(
    BASE_DIR, "processed_data", "curated_users.jsonl"
)

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

def collect_user_book_ratings(users_path, stem_isbns):
    """
    Returns two lists:
        x_values -> number of STEM books read by user
        y_values -> ratings of each STEM book (rating > 0)
    """
    x_values = []
    y_values = []

    with open(users_path, "r", encoding="utf-8") as f:
        for line in f:
            user = json.loads(line)

            # Filter STEM books with rating > 0
            stem_books = [
                book for book in user["book_list"]
                if book["isbn"] in stem_isbns and book["rating"] > 0
            ]

            stem_count = len(stem_books)

            # For each rated STEM book, record (user_stem_count, rating)
            for book in stem_books:
                x_values.append(stem_count)
                y_values.append(book["rating"])

    return x_values, y_values

def main():
    stem_isbns = load_stem_isbns(STEM_BOOKS_PATH)
    x_values, y_values = collect_user_book_ratings(CURATED_USERS_PATH, stem_isbns)

    # ---- Plot ----
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, alpha=0.4)

    plt.xlabel("Number of STEM Books Read by User")
    plt.ylabel("Book Rating")
    plt.title("STEM Book Ratings vs Number of STEM Books Read")

    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()