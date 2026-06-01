import json
import os
import random

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_VECTOR = os.path.join(
    BASE_DIR, "processed_data", "books_with_subjects_read_by_younger_readers.jsonl"
)

STEM_BOOKS = os.path.join(
    BASE_DIR, "processed_data", "stem_isbns_from_classifier.txt"
)

def generate_random_descriptions(txt_path, jsonl_path, output_path, count=300):
    with open(txt_path, "r", encoding="utf-8") as f:
        # strip() removes newlines and whitespace; we store in a set for O(1) lookups
        target_isbns = {line.strip() for line in f if line.strip()}

    matched_books = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                book_data = json.loads(line)
                isbn = book_data.get("ISBN")

                # If this book's ISBN is in our target list, save it
                if isbn in target_isbns:
                    matched_books.append(
                        {
                            "ISBN": isbn,
                            "description": book_data.get("description", ""),
                        }
                    )
            except json.JSONDecodeError:
                # Skips malformed JSON lines if any exist
                continue

    if not matched_books:
        print("Error: No matching ISBNs found between the files.")
        return

    print(f"Found {len(matched_books)} unique matching books available.")

    final_selection = random.sample(matched_books, k=count)

    with open(output_path, "w", encoding="utf-8") as f:
        for book in final_selection:
            f.write(json.dumps(book, ensure_ascii=False) + "\n")

    print(f"Successfully created {output_path} with {count} rows.")


if __name__ == "__main__":
    generate_random_descriptions(STEM_BOOKS, BASE_VECTOR, "sample_stem.jsonl")