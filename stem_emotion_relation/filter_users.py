import json
import os

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CURATED_USERS_PATH = os.path.join(
    BASE_DIR, "processed_data", "curated_users_rated_stem_books.jsonl"
)

STEM_BOOKS_PATH = os.path.join(
    BASE_DIR, "processed_data", "stem_books.jsonl"
)

OUTPUT_PATH_STEM = os.path.join(
    BASE_DIR, "stem_emotion_relation", "any_age_users_who_like_stem_books.jsonl"
)

OUTPUT_PATH_NOT_STEM = os.path.join(
    BASE_DIR, "stem_emotion_relation", "any_age_users_who_do_not_like_stem_books.jsonl"
)

OUTPUT_PATH_MIX = os.path.join(
    BASE_DIR, "stem_emotion_relation", "any_age_users_who_mix_like_stem_books.jsonl"
)

def load_stem_isbns(stem_path):
    """Load all STEM ISBNs into a set for fast lookup."""
    stem_isbns = set()

    with open(stem_path, "r", encoding="utf-8") as f:
        for line in f:
            book = json.loads(line)
            stem_isbns.add(book["ISBN"])

    return stem_isbns


stem_isbns = load_stem_isbns(STEM_BOOKS_PATH)
users_who_like_stem = []
users_who_do_not_like_stem = []
other_users = []

with open(CURATED_USERS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        user = json.loads(line)

        low_rated_stem_books = 0
        high_rate_stem_books = 0
        for book in user["book_list"]:
            if book["isbn"] in stem_isbns and book["rating"] >= 7:
                high_rate_stem_books += 1
            elif book["isbn"] in stem_isbns and book["rating"] > 0  and book["rating"] < 7:
                low_rated_stem_books += 1

        if high_rate_stem_books > 0 and low_rated_stem_books == 0:
            users_who_like_stem.append(user)
        elif low_rated_stem_books > 0 and high_rate_stem_books == 0:
            users_who_do_not_like_stem.append(user)
        elif high_rate_stem_books > 0 and low_rated_stem_books > 0:
            other_users.append(user)
        
print(f"Number of users who like STEM: {len(users_who_like_stem)}")
print(f"Number of users who do not like STEM: {len(users_who_do_not_like_stem)}")
print(f"Number of users who like and dislike STEM: {len(other_users)}")

with open(OUTPUT_PATH_STEM, "w", encoding="utf-8") as f:
    for user in users_who_like_stem:
        f.write(json.dumps(user) + "\n")

with open(OUTPUT_PATH_NOT_STEM, "w", encoding="utf-8") as f:
    for user in users_who_do_not_like_stem:
        f.write(json.dumps(user) + "\n")

with open(OUTPUT_PATH_MIX, "w", encoding="utf-8") as f:
    for user in other_users:
        f.write(json.dumps(user) + "\n")