import json
from Recomender_Helper.vector_helper import get_vector_by_isbn
from tqdm import tqdm

# 1. Load the book metadata from the JSONL file into a lookup dictionary
book_lookup = {}
with open('processed_data/books_with_subjects.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        book_data = json.loads(line)
        isbn = book_data.get("ISBN")
        if isbn:
            book_lookup[isbn] = {
                "title": book_data.get("Book-Title", "Unknown Title"),
                "author": book_data.get("Book-Author", "Unknown Author")
            }

# 2. Load your main user profiles JSON file
with open('user_eval_sets/users_1_plus_STEM_books_and_10_plus_high_rated_split_60_40.json', 'r', encoding='utf-8') as f:
    users = json.load(f)

# 3. Enrich the data
def enrich_book_list(book_list):
    for book in book_list:
        isbn = book.get("isbn")
        e_vec = get_vector_by_isbn(isbn, "emotion_intensity")
        t_vec = get_vector_by_isbn(isbn, "empath")
        book["emotion_vector"] = e_vec
        book["topic_vector"] = t_vec
        if isbn in book_lookup:
            book["title"] = book_lookup[isbn]["title"]
            book["author"] = book_lookup[isbn]["author"]
        else:
            book["title"] = "Metadata not found"
            book["author"] = "Metadata not found"

for user in tqdm(users):
    enrich_book_list(user.get("candidate_profile", []))
    enrich_book_list(user.get("recommendation_list", []))

# 4. Save the enriched data to a new JSON file
with open('enriched_user_profiles.json', 'w', encoding='utf-8') as f:
    json.dump(users, f, indent=4)

print("Enrichment complete! Saved to enriched_user_profiles.json")