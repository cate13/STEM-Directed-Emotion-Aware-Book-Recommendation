import json
import math
import os
import random
from collections import Counter

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


def process_book_data(input_file, output_file, stem_isbns):
    results = []

    with open(input_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            
            data = json.loads(line)
            book_list = data.get('book_list', [])
            
            highly_rated_allowed = [
                book for book in book_list 
                if book['rating'] >= 7 and book['isbn'] not in stem_isbns
            ]
            
            random.shuffle(highly_rated_allowed)
            
            split_idx = math.ceil(len(highly_rated_allowed) / 2)
            candidate_profile = highly_rated_allowed[:split_idx]
            
            profile_isbns = {b['isbn'] for b in candidate_profile}

            recommendation_list = []

            for book in book_list:
                if book['isbn'] not in profile_isbns:
                    if book['isbn'] in stem_isbns:
                        book['is_stem'] = True
                    
                    recommendation_list.append(book)
            
            new_entry = {
                "user_id": data['user_id'],
                "age": data['age'],
                "candidate_profile": candidate_profile,
                "recommendation_list": recommendation_list
            }
            results.append(new_entry)

    # Save to a standard JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

stem_isbn = load_stem_isbns(STEM_BOOKS_PATH)

random.seed(42)

process_book_data('user_eval_sets/users_with_1_2_3_STEM_books_and_six_plus_high_rated_books.jsonl', 
                'user_eval_sets/users_with_1_2_3_STEM_books_and_six_plus_high_rated_books_formatted.json', stem_isbn)