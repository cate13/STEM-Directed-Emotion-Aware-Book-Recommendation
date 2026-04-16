import json
import os
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSONL_PATH_1 = os.path.join(BASE_DIR, "processed_data", "books_with_subjects_1.jsonl")
JSONL_PATH_2 = os.path.join(BASE_DIR, "processed_data", "books_with_subjects_2.jsonl")
JSONL_PATH_OUT = os.path.join(BASE_DIR, "processed_data", "books_with_subjects_combined.jsonl")

def merge_jsonl_files(file1_path, file2_path, output_path):
    seen_isbns = set()
    
    with open(output_path, 'w', encoding='utf-8') as outfile:
        # Process the first file: Add everything and track ISBNs
        with open(file1_path, 'r', encoding='utf-8') as f1:
            for line in tqdm(f1):
                if not line.strip():
                    continue
                data = json.loads(line)
                isbn = data.get("ISBN")
                if isbn:
                    seen_isbns.add(isbn)
                outfile.write(line.strip() + '\n')
        
        # Process the second file: Only add if ISBN is new
        with open(file2_path, 'r', encoding='utf-8') as f2:
            for line in tqdm(f2):
                if not line.strip():
                    continue
                data = json.loads(line)
                isbn = data.get("ISBN")
                
                if isbn not in seen_isbns:
                    outfile.write(line.strip() + '\n')
                    seen_isbns.add(isbn) # Optional: prevents dupes within file2 itself

# Usage
merge_jsonl_files(JSONL_PATH_1, JSONL_PATH_2, JSONL_PATH_OUT)