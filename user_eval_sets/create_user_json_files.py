import json
import os
import math
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CURATED_USERS_PATH = os.path.join(
    BASE_DIR, "processed_data", "curated_users_12-25.jsonl"
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

DESCRIPTION_BOOKS = os.path.join(
    BASE_DIR, "processed_data", "curated_users.jsonl"
)

def load_books_with_description_isbn():
    isbns = set()

    with open(DESCRIPTION_BOOKS, "r", encoding="utf-8") as f:
        for line in f:
            isbns.add(line.strip())

    return isbns

def load_stem_isbns(stem_paths):
    """Load all STEM ISBNs into a set for fast lookup."""
    stem_isbns = set()

    for path in stem_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                stem_isbns.add(line.strip())

    return stem_isbns

def filter_users(file_path, target_isbns, description_isbns, x, y = 4):
    """
    Finds users who have:
    1. At least x books rated > 7
    2. At least y books found in the target_isbns set
    """
    matching_users = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Skip empty lines if any
            if not line.strip():
                continue
            
            user_data = json.loads(line)
            book_list = user_data.get('book_list', [])
            
            # Counters for the specific user
            high_rating_count = 0
            isbn_match_count = 0
            # print(f"user {user_data.get('user_id')}: {book_list}")
            for book in book_list:
                isbn = book.get('isbn')

                if isbn in description_isbns:
                    # Check rating criteria (above 7)
                    if book.get('rating', 0) >= 7:
                        high_rating_count += 1
                    
                    # Check if ISBN is in the target set
                    if isbn in target_isbns:
                        # print(f"\t {isbn} is STEM")
                        isbn_match_count += 1
            
            # Apply both filters
            #if high_rating_count >= x and isbn_match_count >= y:
            if high_rating_count >= x and isbn_match_count > 1 and isbn_match_count < 4:
                matching_users.append(user_data)
            
                
    return matching_users

def format_matching_users(matching_users, target_isbns):
    formatted_results = []

    for user in matching_users:
        all_books = user.get('book_list', [])
        
        # 1. Identify "Stem" books and "Candidate-Eligible" books
        # Candidate-Eligible = Rating > 7 AND NOT in target_isbns
        candidate_eligible = []
        others = []

        for book in all_books:
            book_copy = book.copy()
            is_in_target = book_copy.get('isbn') in target_isbns
            
            if is_in_target:
                book_copy['is_stem'] = True
                others.append(book_copy)
            elif book_copy.get('rating', 0) > 7:
                candidate_eligible.append(book_copy)
            else:
                others.append(book_copy)

        # 2. Split the candidate_eligible books into two halves
        random.shuffle(candidate_eligible)
        split_point = int(len(candidate_eligible) * 0.6)
        
        candidate_profile = candidate_eligible[:split_point]
        
        # 3. recommendation_list gets:
        # - The second half of the high-rated non-target books
        # - All the target_isbns (stem books)
        # - All low-rated books
        recommendation_list = candidate_eligible[split_point:] + others

        # 4. Construct the final object
        formatted_user = {
            "user_id": user.get("user_id"),
            "age": user.get("age"),
            "candidate_profile": candidate_profile,
            "recommendation_list": recommendation_list
        }
        
        formatted_results.append(formatted_user)

    return formatted_results


def format_matching_users_for_STEM_in_profile(matching_users, target_isbns):
    formatted_results = []

    for user in matching_users:
        all_books = user.get('book_list', [])
        
        # 1. Identify "Stem" books and "Candidate-Eligible" books
        # Candidate-Eligible = Rating > 7 AND NOT in target_isbns
        candidate_eligible = []
        stem_books = []
        others = []

        for book in all_books:
            book_copy = book.copy()
            is_in_target = book_copy.get('isbn') in target_isbns
            
            if is_in_target:
                book_copy['is_stem'] = True
                stem_books.append(book_copy)
            elif book_copy.get('rating', 0) > 7:
                candidate_eligible.append(book_copy)
            else:
                others.append(book_copy)

        stem_books.sort(key=lambda x: x['rating'], reverse=True)
        stem_profile = stem_books[0::2]
        stem_other = stem_books[1::2]

        # 2. Split the candidate_eligible books into two halves
        random.shuffle(candidate_eligible)
        split_point = len(candidate_eligible) // 2
        
        candidate_profile = candidate_eligible[:split_point] + stem_profile


        
        # 3. recommendation_list gets:
        # - The second half of the high-rated non-target books
        # - All the target_isbns (stem books)
        # - All low-rated books
        recommendation_list = candidate_eligible[split_point:] + others + stem_other

        # 4. Construct the final object
        formatted_user = {
            "user_id": user.get("user_id"),
            "age": user.get("age"),
            "candidate_profile": candidate_profile,
            "recommendation_list": recommendation_list
        }
        
        formatted_results.append(formatted_user)

    return formatted_results

def get_users_with_1_2_3_STEM_books(stem_isbns):
    matching_users = []

    with open(CURATED_USERS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            # Skip empty lines if any
            if not line.strip():
                continue
            
            user_data = json.loads(line)
            book_list = user_data.get('book_list', [])
            isbn_match_count = 0
            for book in book_list:
                isbn = book.get('isbn')
                if isbn in stem_isbns:
                        # print(f"\t {isbn} is STEM")
                        isbn_match_count += 1
            if isbn_match_count > 0 and isbn_match_count < 4:
                matching_users.append(user_data)
    return matching_users

def get_users_with_at_least_1_STEM_book(stem_isbns):
    matching_users = []

    with open(CURATED_USERS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            user_data = json.loads(line)
            book_list = user_data.get('book_list', [])
            
            for book in book_list:
                isbn = book.get('isbn')
                if isbn in stem_isbns:
                    matching_users.append(user_data)
                    # Use break to stop checking books for THIS user
                    break 

    return matching_users



def get_users_with_4_plus_STEM_books(stem_isbns):
    matching_users = []

    with open(CURATED_USERS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            # Skip empty lines if any
            if not line.strip():
                continue
            
            user_data = json.loads(line)
            book_list = user_data.get('book_list', [])
            isbn_match_count = 0
            for book in book_list:
                isbn = book.get('isbn')
                if isbn in stem_isbns:
                        # print(f"\t {isbn} is STEM")
                        isbn_match_count += 1
            if isbn_match_count >= 4:
                matching_users.append(user_data)
    return matching_users

def filter_for_6_highly_rated_books(starting_users):
    matching_users = []

    for user in starting_users:
        book_list = user.get('book_list', [])
        high_rating_count = 0
        for book in book_list:
            if book.get('rating', 0) >= 7:
                high_rating_count += 1

        if high_rating_count >= 6:
            matching_users.append(user)
                 
    return matching_users

def filter_for_4_highly_rated_books(starting_users):
    matching_users = []

    for user in starting_users:
        book_list = user.get('book_list', [])
        high_rating_count = 0
        for book in book_list:
            if book.get('rating', 0) >= 7:
                high_rating_count += 1

        if high_rating_count >= 4:
            matching_users.append(user)
                 
    return matching_users       

def filter_for_x_high_rated_books(starting_users, stem_books, book_count):
    matching_users = []

    for user in starting_users:
        book_list = user.get('book_list', [])
        high_rating_count = 0
        for book in book_list:
            isbn = book.get('isbn')
            if isbn in stem_books: continue 
            rating = int(book.get('rating', 0)) 
            if rating >= 7:
                high_rating_count += 1

        if high_rating_count >= book_count:
            matching_users.append(user)
                 
    return matching_users   

def get_60_40_split(highly_rated_book_count = 6):
    stem_isbns = load_stem_isbns([STEM_BOOKS_PATH_1, STEM_BOOKS_PATH_2, STEM_BOOKS_PATH_3])
    users = get_users_with_at_least_1_STEM_book(stem_isbns)
    user_with_x = filter_for_x_high_rated_books(users, stem_isbns, highly_rated_book_count)
    print(len(user_with_x))

    formatted_users = format_matching_users(user_with_x, stem_isbns)

    out_file_path = os.path.join(
        BASE_DIR, "user_eval_sets", f"users_1_plus_STEM_books_and_{highly_rated_book_count}_plus_high_rated_split_60_40.json"
    )

    with open(out_file_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_users, f, indent=4, ensure_ascii=False)
        
    print(f"Successfully saved {len(formatted_users)} users to {out_file_path}")

def get_1_2_3_user_list_step_by_step():
    stem_isbns = load_stem_isbns([STEM_BOOKS_PATH_1, STEM_BOOKS_PATH_2, STEM_BOOKS_PATH_3])
    # get users with 1-2-3 STEM books:
    users_with_1_2_3_STEM_books = get_users_with_1_2_3_STEM_books(stem_isbns)
    print(len(users_with_1_2_3_STEM_books))

    #get users with 6+ highly rated books:
    users_with_1_2_3_STEM_books_and_6_plus_high_rating = filter_for_6_highly_rated_books(users_with_1_2_3_STEM_books)
    print(len(users_with_1_2_3_STEM_books_and_6_plus_high_rating))

    formatted_users = format_matching_users(users_with_1_2_3_STEM_books_and_6_plus_high_rating, stem_isbns)

    out_file_path = os.path.join(
        BASE_DIR, "user_eval_sets", "users_1_2_3_STEM_books_and_6_plus_high_rated.json"
    )

    with open(out_file_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_users, f, indent=4, ensure_ascii=False)
        
    print(f"Successfully saved {len(formatted_users)} users to {out_file_path}")

def get_4_plus_user_list():
    stem_isbns = load_stem_isbns([STEM_BOOKS_PATH_1, STEM_BOOKS_PATH_2, STEM_BOOKS_PATH_3])
    users_with_4_plus_stem_books = get_users_with_4_plus_STEM_books(stem_isbns)
    print(len(users_with_4_plus_stem_books))

    users_with_4_plus_stem_books_and_4_plus_high_rating = filter_for_4_highly_rated_books(users_with_4_plus_stem_books)
    print(len(users_with_4_plus_stem_books_and_4_plus_high_rating))

    formatted_users = format_matching_users_for_STEM_in_profile(users_with_4_plus_stem_books_and_4_plus_high_rating, stem_isbns)

    out_file_path = os.path.join(
        BASE_DIR, "user_eval_sets", "users_4_plus_STEM_books_and_6_plus_high_rated.json"
    )

    with open(out_file_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_users, f, indent=4, ensure_ascii=False)
        
    print(f"Successfully saved {len(formatted_users)} users to {out_file_path}")

get_60_40_split(highly_rated_book_count=6)