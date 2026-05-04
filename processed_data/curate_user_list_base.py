import csv
import json
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USERS_PATH = os.path.join(BASE_DIR, "starting_data", "Users.csv")
RATINGS_PATH = os.path.join(BASE_DIR, "starting_data", "Ratings.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "processed_data", "curated_users.jsonl")

def load_users(users_path, start_age = 12, end_age = 20):
    """Load users between ages 12 and 20."""
    valid_users = {}

    with open(users_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            age = row["Age"].strip()

            if age == "":
                continue

            try:
                age = float(age)
            except ValueError:
                continue

            if start_age <= age <= end_age:
                user_id = int(row["User-ID"])
                valid_users[user_id] = {
                    "user_id": user_id,
                    "age": age,
                    "book_list": []
                }

    return valid_users

def attach_ratings(ratings_path, valid_users):
    """Attach book ratings to valid users."""
    with open(ratings_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_id = int(row["User-ID"])

            if user_id in valid_users:
                valid_users[user_id]["book_list"].append({
                    "isbn": row["ISBN"],
                    "rating": int(row["Book-Rating"])
                })

def write_jsonl(output_path, users_dict):
    """Write users to JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for user in users_dict.values():
            f.write(json.dumps(user) + "\n")

def main():
    valid_users = load_users(USERS_PATH, end_age=25)
    print(len(valid_users))
    attach_ratings(RATINGS_PATH, valid_users)
    write_jsonl(OUTPUT_PATH, valid_users)
    print(f"Saved curated users to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()