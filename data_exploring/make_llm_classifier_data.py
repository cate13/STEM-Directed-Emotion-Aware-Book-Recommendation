import json
from pathlib import Path

def filter_books(stem_path, file2_path, input_jsonl, output_jsonl):
    # 1. Load ISBNs into sets for fast lookup
    with open(stem_path, 'r') as f:
        # Using strip() to remove newlines and whitespace
        stem_isbns = set(line.strip() for line in f if line.strip())
        
    with open(file2_path, 'r') as f:
        isbns_file2 = set(line.strip() for line in f if line.strip())

    # Combine sets to identify which ISBNs we care about at all
    all_target_isbns = stem_isbns.union(isbns_file2)

    with open(input_jsonl, 'r', encoding='utf-8') as infile, \
         open(output_jsonl, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            try:
                book_data = json.loads(line)
                isbn = book_data.get("ISBN")

                # 2. Check if the ISBN exists in our target lists
                if isbn in all_target_isbns:
                    # 3. Construct the new filtered object
                    filtered_entry = {
                        "ISBN": isbn,
                        "Book-Title": book_data.get("Book-Title"),
                        "Book-Author": book_data.get("Book-Author"),
                        "description": book_data.get("description"),
                        # Set boolean: True if in file1, False otherwise
                        "is_stem": isbn in stem_isbns 
                    }
                    
                    # 4. Write to the new JSONL file
                    outfile.write(json.dumps(filtered_entry) + '\n')
            except json.JSONDecodeError:
                continue

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    stem_txt_path = base_dir.parent / "data_exploring" / "stem_books_read_by_youth.txt"
    non_stem_path = base_dir.parent / "data_exploring" / "750_non_stem_books_read_by_youth_with_descriptions.txt"
    description_books = base_dir.parent / "processed_data" / "books_with_subjects_complete.jsonl"
    output_path = base_dir.parent / "Classifier" / "books_for_llm_classification.jsonl"


    filter_books(
        stem_path=stem_txt_path, 
        file2_path=non_stem_path, 
        input_jsonl=description_books, 
        output_jsonl=output_path
    )