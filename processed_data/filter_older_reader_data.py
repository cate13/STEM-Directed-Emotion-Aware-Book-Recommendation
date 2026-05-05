import json
import os

def filter_stem_books_to_file(jsonl_path, topics_path, output_path):
    # 1. Load STEM topics into a set for O(1) lookup speed
    with open(topics_path, 'r', encoding='utf-8') as f:
        # Strip whitespace and quotes if they exist in the text file
        stem_topics = {line.strip().strip('"') for line in f if line.strip()}

    count = 0

    # 2. Open the output file for writing
    with open(output_path, 'w', encoding='utf-8') as out_file:
        # 3. Process the JSONL file line by line
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                book = json.loads(line)
                
                # Extract lists, defaulting to empty list if key is missing
                loc_subjects = book.get("LoC_subjects", [])
                google_cats = book.get("Google_categories", [])
                
                # Combine both lists into a set for comparison
                combined_topics = set(loc_subjects) | set(google_cats)
                
                # Check for intersection between book topics and STEM topics
                if combined_topics.intersection(stem_topics):
                    isbn = book.get("ISBN")
                    if isbn:
                        out_file.write(f"{isbn}\n")
                        count += 1

    print(f"Extraction complete. {count} ISBNs written to {output_path}")

# --- Execution ---
jsonl_input = "processed_data/books_with_subjects_read_by_older_readers.jsonl"
topics_input = "STEM_Books/STEM_topics.txt"
output_file = "STEM_ISBNS_from_older_reader.txt"

filter_stem_books_to_file(jsonl_input, topics_input, output_file)