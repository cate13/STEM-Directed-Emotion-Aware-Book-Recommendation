import json

# Files
USER_FILE = "user_eval_sets/users_emotion_only_where_high_rated_is_8.json"
EMOTION_FILE = "processed_data/book_vectors_base_younger_readers.jsonl"
BOOKS_FILE = "processed_data/books_with_subjects_read_by_younger_readers.jsonl"
OUTPUT_FILE = "users_enriched_rated_8.json"


# Load emotion data
emotion_lookup = {}
with open(EMOTION_FILE, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        emotion_lookup[record["isbn"]] = record

# Load book metadata
book_lookup = {}
with open(BOOKS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        book_lookup[record["ISBN"]] = record

# Load user data
with open(USER_FILE, "r", encoding="utf-8") as f:
    users = json.load(f)


def enrich_book(book):
    isbn = book["isbn"]

    emotion_record = emotion_lookup.get(isbn)
    metadata = book_lookup.get(isbn)

    # Require both records to exist
    if emotion_record is None or metadata is None:
        return None

    # Require title and author
    title = metadata.get("Book-Title")
    author = metadata.get("Book-Author")

    if not title or not author:
        return None

    # Require emotion_intensity
    emotion_intensity = emotion_record.get("emotion_intensity")
    if emotion_intensity is None:
        return None

    return {
        **book,
        "emotion_intensity": emotion_intensity,
        "title": title,
        "author": author,
    }


# Enrich all books
for user in users:
    # Candidate profile
    candidate_profile = []
    for book in user.get("candidate_profile", []):
        enriched = enrich_book(book)
        if enriched is not None:
            candidate_profile.append(enriched)
    user["candidate_profile"] = candidate_profile

    # Recommendation list
    recommendation_list = []
    for book in user.get("recommendation_list", []):
        enriched = enrich_book(book)
        if enriched is not None:
            recommendation_list.append(enriched)
    user["recommendation_list"] = recommendation_list

# Save result
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(users, f, indent=4, ensure_ascii=False)

print(f"Saved {len(users)} users to {OUTPUT_FILE}")