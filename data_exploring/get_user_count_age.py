import pandas as pd
import glob
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

users_path = os.path.join(
    BASE_DIR, "starting_data", 'Users.csv'
)

reviews_path = os.path.join(
    BASE_DIR, "starting_data", 'Ratings.csv'
)

isbn_folder = os.path.join(
    BASE_DIR, "processed_data"
)

try:
    users = pd.read_csv(users_path)
    reviews = pd.read_csv(reviews_path)
except FileNotFoundError as e:
    print(f"Error: Could not find files. Make sure you are running the script from within folder_1. {e}")
    exit()

users['Age'] = pd.to_numeric(users['Age'], errors='coerce')

isbn_files = glob.glob(os.path.join(isbn_folder, "*.txt"))

stem_isbns = set()
for file_path in isbn_files:
    with open(file_path, 'r') as f:
        isbns = [line.strip() for line in f.readlines() if line.strip()]
        stem_isbns.update(isbns)

user_book_counts = reviews.groupby('User-ID').size()
users_with_stem_books = reviews[reviews['ISBN'].isin(stem_isbns)]['User-ID'].unique()

high_ratings = reviews[reviews['Book-Rating'] >= 7]
# Count how many high-rated books each user has
high_rating_counts = high_ratings.groupby('User-ID').size()
# Users with 10+ books rated 7+
users_10_high_ratings = set(high_rating_counts[high_rating_counts >= 10].index)

special_category_users = users_10_high_ratings.intersection(users_with_stem_books)

def get_stats(min_age, max_age):
    mask = (users['Age'] >= min_age) & (users['Age'] <= max_age)
    range_users = users[mask]
    range_user_ids = range_users['User-ID']
    
    total_count = len(range_users)

    read_1_plus = range_user_ids[range_user_ids.isin(user_book_counts[user_book_counts >= 1].index)].count()
    
    # Users in range who read at least 3 books
    reads_3_plus = range_user_ids[range_user_ids.isin(user_book_counts[user_book_counts >= 3].index)].count()
    
    # Users in range who read at least 1 STEM book
    reads_stem = range_user_ids[range_user_ids.isin(users_with_stem_books)].count()

    special_count = range_user_ids[range_user_ids.isin(list(special_category_users))].count()
    
    return total_count, read_1_plus, reads_3_plus, reads_stem, special_count

ranges = [(12, 19), (12, 20), (12, 21), (12, 22), (12, 22), (12, 24), (12, 25)]

# print(f"{'Age Range':<12} | {'Total Users':<12} | {'Read 1+ Books':<15} | {'Read 3+ Books':<15} | {'at least 1 STEM':<20} | {'10+ Highly Rated and 1+ STEM':<20}") 
# print("-" * 65)

# for start, end in ranges:
#     total, count1, count3, count_stem, special_count = get_stats(start, end)
#     print(f"{start}-{end:<9} | {total:<12} | {count1:<15} | {count3:<15} | {count_stem:<20} | {special_count:<20}")

results_data = []

for start, end in ranges:
    total, count1, count3, count_stem, special_count = get_stats(start, end)
    
    results_data.append({
        'Age Range': f"{start}-{end}",
        'Total Users': total,
        'Read 1+': count1,
        'Read 3+': count3,
        'Read 1+ STEM': count_stem,
        '10+ High Rated and 1+ STEM': special_count
    })

df_output = pd.DataFrame(results_data)
df_output.to_csv('age_range_stats.csv', index=False)

print("Data successfully saved to age_range_stats.csv")