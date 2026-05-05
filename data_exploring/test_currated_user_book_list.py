import json
import os

# Update this path to match your actual file location
CURATED_USERS_PATH = "processed_data/curated_users_12-25.jsonl"
VECTORED_ISBNS_PATH = "processed_data/book_vectors_base.jsonl"

def extract_unique_isbns_from_book_vectors_base(file_path):
    unique_isbns = set()

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            isbn = data.get("isbn")
            if isbn:
                unique_isbns.add(str(isbn).strip())

    return unique_isbns

def extract_unique_isbns_from_currated_users(file_path):
    unique_isbns = set()
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return unique_isbns

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                # Iterate through the list of book objects for this user
                for book in data.get("book_list", []):
                    isbn = book.get("isbn")
                    if isbn:
                        unique_isbns.add(str(isbn).strip())
            except json.JSONDecodeError:
                print(f"Skipping malformed JSON on line {line_num}")
                
    return unique_isbns

# Usage
all_isbns_from_curated_users = extract_unique_isbns_from_currated_users(CURATED_USERS_PATH)
all_isbns_from_vector = extract_unique_isbns_from_book_vectors_base(VECTORED_ISBNS_PATH)

missing_vectors = all_isbns_from_curated_users - all_isbns_from_vector

print(f"Number of books missing vectors: {len(missing_vectors)}")


# Quick check: is a specific problematic ISBN in this set?
# test_isbn = "0553571737"
# if test_isbn in all_isbns:
#     print(f"Target ISBN {test_isbn} IS present in the curated file.")
# else:
#     print(f"Target ISBN {test_isbn} is NOT in the curated file.")