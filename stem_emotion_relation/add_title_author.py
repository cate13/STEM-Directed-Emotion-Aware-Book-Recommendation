import json
import os
import numpy as np

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATH_STEM = os.path.join(
    BASE_DIR, "stem_emotion_relation", "any_age_users_who_like_stem_books_with_vec.jsonl"
)

PATH_NOT_STEM = os.path.join(
    BASE_DIR, "stem_emotion_relation", "any_age_users_who_do_not_like_stem_books_with_vec.jsonl"
)

PATH_MIX = os.path.join(
    BASE_DIR, "stem_emotion_relation", "any_age_users_who_mix_like_stem_books_with_vec.jsonl"
)

STEM_BOOKS_PATH = os.path.join(
    BASE_DIR, "processed_data", "stem_books.jsonl"
)

ALL_BOOKS_PATH = os.path.join(
    BASE_DIR, "processed_data", "books_with_subjects_complete.jsonl"
)

def load_stem_isbns(stem_path):
    """Load all STEM ISBNs into a set for fast lookup."""
    stem_isbns = set()

    with open(stem_path, "r", encoding="utf-8") as f:
        for line in f:
            book = json.loads(line)
            stem_isbns.add(book["ISBN"])

    return stem_isbns

def load_book_metadata():
    book_metadata = {}
    with open(ALL_BOOKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            book_metadata[data["ISBN"]] = {
                "title": data.get("Book-Title", "Unknown"),
                "author": data.get("Book-Author", "Unknown")
            }
    return book_metadata

book_metadata = load_book_metadata()
stem_isbns = load_stem_isbns(STEM_BOOKS_PATH)

def process_user_file(input_path, output_path, stem_set, metadata):
    """Processes a single jsonl file and writes to a new one."""
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            user_data = json.loads(line)
            
            for book in user_data.get("book_list", []):
                isbn = book.get("isbn")
                
                # Check for STEM status
                if isbn in stem_set:
                    book["is_stem"] = True
                
                # Enrich with Title and Author
                info = metadata.get(isbn, {"title": "Unknown", "author": "Unknown"})
                book["title"] = info["title"]
                book["author"] = info["author"]
            
            # Write the updated user object back as a line
            f_out.write(json.dumps(user_data) + "\n")

# 2. Map your paths and run the loop
files_to_process = {
    PATH_STEM: PATH_STEM.replace(".jsonl", "_enriched.jsonl"),
    PATH_NOT_STEM: PATH_NOT_STEM.replace(".jsonl", "_enriched.jsonl"),
    PATH_MIX: PATH_MIX.replace(".jsonl", "_enriched.jsonl"),
}

for inp, outp in files_to_process.items():
    print(f"Processing {os.path.basename(inp)}...")
    process_user_file(inp, outp, stem_isbns, book_metadata)

print("Done!")