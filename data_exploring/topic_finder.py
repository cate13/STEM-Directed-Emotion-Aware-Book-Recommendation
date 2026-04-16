import json
from collections import Counter

def extract_counted_values(input_file):
    loc_subjects = Counter()
    google_categories = Counter()

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line.strip())
                
                subjects = data.get('LoC_subjects', [])
                categories = data.get('Google_categories', [])
                
                if isinstance(subjects, list):
                    loc_subjects.update(subjects)
                
                if isinstance(categories, list):
                    google_categories.update(categories)

        # Helper to write ONLY the names, sorted by frequency
        def save_ranked_names(counter_obj, filename):
            with open(filename, 'w', encoding='utf-8') as f_out:
                # We iterate through the tuples, but only use 'item'
                for item, count in counter_obj.most_common():
                    f_out.write(f"{item}\n")

        save_ranked_names(loc_subjects, 'loc_subjects.txt')
        save_ranked_names(google_categories, 'google_categories.txt')

        print("Done! Files created. Most frequent items are at the top.")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON. Ensure the file is correctly formatted JSONL.")

# Run the function
if __name__ == "__main__":
    # Replace 'books.jsonl' with your actual filename
    extract_counted_values('processed_data/books_with_subjects.jsonl')