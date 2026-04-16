import csv
import json
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USERS_PATH = os.path.join(BASE_DIR, "starting_data", "Users.csv")
RATINGS_PATH = os.path.join(BASE_DIR, "starting_data", "Ratings.csv")
BOOKS_JSONL_PATH = os.path.join(BASE_DIR, "processed_data", "books_with_subjects.jsonl") 
OUTPUT_PATH = os.path.join(BASE_DIR, "processed_data", "curated_users.jsonl")

def load_valid_isbns(books_path):
    """Create a set of ISBNs that exist in the books JSONL."""
    valid_isbns = set()
    if not os.path.exists(books_path):
        print(f"Warning: {books_path} not found.")
        return valid_isbns
        
    with open(books_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                book_data = json.loads(line)
                valid_isbns.add(book_data["ISBN"])
            except (json.JSONDecodeError, KeyError):
                continue
    return valid_isbns

def load_users(users_path):
    """Load users between ages 12 and 20."""
    valid_users = {}
    with open(users_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            age = row["Age"].strip()
            if age == "" or not age: continue
            try:
                age = float(age)
            except ValueError: continue
            
            if 12 <= age <= 20:
                u_id = int(row["User-ID"])
                valid_users[u_id] = {"user_id": u_id, "age": age, "book_list": []}
    return valid_users

def attach_ratings(ratings_path, valid_users, valid_isbns):
    """Attach ratings if user is valid, rating > 0, and ISBN exists in book metadata."""
    with open(ratings_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u_id = int(row["User-ID"])
            isbn = row["ISBN"]
            rating = int(row["Book-Rating"])

            # Filter logic: User exists AND Rating > 0 AND ISBN exists in our JSONL set
            if u_id in valid_users and rating > 0 and isbn in valid_isbns:
                valid_users[u_id]["book_list"].append({
                    "isbn": isbn,
                    "rating": rating
                })

def write_jsonl(output_path, users_dict):
    """Write users only if they ended up with books in their list."""
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for user in users_dict.values():
            if len(user["book_list"]) > 0:
                f.write(json.dumps(user) + "\n")
                count += 1
    return count

def main():
    print("Loading valid ISBNs...")
    valid_isbns = load_valid_isbns(BOOKS_JSONL_PATH)
    print(f"Valid books: {len(valid_isbns)}")
    
    print("Loading users...")
    valid_users = load_users(USERS_PATH)
    print(f"valid users: {len(valid_users)}")
    
    print("Processing ratings...")
    attach_ratings(RATINGS_PATH, valid_users, valid_isbns)
    
    print("Writing curated data...")
    final_count = write_jsonl(OUTPUT_PATH, valid_users)
    
    print(f"Done! Saved {final_count} users to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()