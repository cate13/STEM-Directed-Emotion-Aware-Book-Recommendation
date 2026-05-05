import json
import os
from tqdm import tqdm 
from Recomender_Helper.vector_helper import get_vector_by_isbn

# Update this path to match your actual file location
CURATED_USERS_PATH = "processed_data/curated_users_12-25.jsonl"


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
print(len(all_isbns_from_curated_users))

missing_isbn = 0

for isbn in tqdm(all_isbns_from_curated_users):
    try:
        emo_lex = get_vector_by_isbn(isbn, "emotion_intensity")
        nrc = get_vector_by_isbn(isbn, "emotion")
        empath = get_vector_by_isbn(isbn, "empath")
        tf_idf = get_vector_by_isbn(isbn, "tf_idf")

        if any(v is None for v in [emo_lex, nrc, empath, tf_idf]):
            #print(f"Skipping ISBN {isbn}: One or more vectors are missing.")
            missing_isbn += 1
            continue
    except Exception as e:
        print(e)
        print(isbn)

print(f"Missing {missing_isbn} isbns")