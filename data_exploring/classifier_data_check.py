import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SBERT_VECTOR_FILE_PATH = os.path.join(BASE_DIR, "Classifier", "empath_7d_sentence_bert_long_vectors.jsonl")
ALL_THE_REST_VECTOR_FILE_PATH = os.path.join(BASE_DIR, "Classifier", "book_vectors_trainging_classifier.jsonl")
OUTPUT_PATH = os.path.join(BASE_DIR, "Classifier", "book_vectors_trainging_classifier_include_sbert.jsonl")


def get_isbns_from_jsonl(file_path):
    """Extracts all ISBNs from a JSONL file into a set."""
    isbns = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    isbn = data.get('isbn')
                    if isbn:
                        isbns.add(str(isbn))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON on line {line_num} in {file_path}")
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    return isbns

def compare_jsonl_isbns(file1, file2):
    isbns1 = get_isbns_from_jsonl(file1)
    isbns2 = get_isbns_from_jsonl(file2)

    if isbns1 is None or isbns2 is None:
        return

    print(f"File 1 total unique ISBNs: {len(isbns1)}")
    print(f"File 2 total unique ISBNs: {len(isbns2)}")

    if isbns1 == isbns2:
        print("\n✅ Success: Both files contain the exact same set of ISBNs.")
    else:
        print("\n❌ Difference found: The ISBN sets are not identical.")
        
        only_in_1 = isbns1 - isbns2
        only_in_2 = isbns2 - isbns1
        
        if only_in_1:
            print(f"ISBNs only in the first file ({len(only_in_1)}): {list(only_in_1)[:10]}...")
        if only_in_2:
            print(f"ISBNs only in the second file ({len(only_in_2)}): {list(only_in_2)[:10]}...")

def merge_jsonl_files(file1_path, file2_path, output_path):
    # We'll store data from the first file in a dictionary keyed by ISBN
    data_map = {}

    print(f"Reading {file1_path}...")
    with open(file1_path, 'r', encoding='utf-8') as f1:
        for line in f1:
            if line.strip():
                item = json.loads(line)
                isbn = item.get('isbn')
                if isbn:
                    data_map[isbn] = item

    print(f"Merging with {file2_path} and writing to {output_path}...")
    with open(file2_path, 'r', encoding='utf-8') as f2, \
         open(output_path, 'w', encoding='utf-8') as out_f:
        
        for line in f2:
            if line.strip():
                item2 = json.loads(line)
                isbn = item2.get('isbn')
                
                if isbn in data_map:
                    # Get the data from the first file
                    item1 = data_map[isbn]
                    
                    # Merge all keys from both dictionaries
                    # Dictionary unpacking (**) handles the merging
                    # item2 keys will overwrite item1 keys if they overlap
                    merged_record = {**item1, **item2}
                    
                    # Write the merged result to the new file
                    out_f.write(json.dumps(merged_record) + '\n')
                else:
                    print(f"Warning: ISBN {isbn} found in file 2 but not in file 1. Skipping.")

    print("Merge complete!")

merge_jsonl_files(SBERT_VECTOR_FILE_PATH, ALL_THE_REST_VECTOR_FILE_PATH, OUTPUT_PATH)
#compare_jsonl_isbns(SBERT_VECTOR_FILE_PATH, ALL_THE_REST_VECTOR_FILE_PATH)