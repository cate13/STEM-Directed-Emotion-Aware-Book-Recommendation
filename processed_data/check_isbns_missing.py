import json

def find_missing_isbns(user_file, subjects_file, vectors_file):
    user_isbns = set()
    subjects_isbns = set()
    vectors_isbns = set()

    # 1. Collect all ISBNs from curated_users.jsonl
    print("Reading curated_users.jsonl...")
    with open(user_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            for book in data.get('book_list', []):
                isbn = book.get('isbn')
                if isbn:
                    user_isbns.add(isbn)

    # 2. Collect all ISBNs from books_with_subjects.jsonl
    print("Reading books_with_subjects.jsonl...")
    with open(subjects_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            isbn = data.get('ISBN')  # Note: Uppercase in this file
            if isbn:
                subjects_isbns.add(isbn)

    # 3. Collect all ISBNs from book_vectors_base.jsonl
    print("Reading book_vectors_base.jsonl...")
    with open(vectors_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            isbn = data.get('isbn')  # Note: Lowercase in this file
            if isbn:
                vectors_isbns.add(isbn)

    # 4. Calculate sets of missing ISBNs
    missing_from_subjects = user_isbns - subjects_isbns
    missing_from_vectors = user_isbns - vectors_isbns
    
    # ISBNs missing from BOTH files
    missing_from_both = user_isbns - (subjects_isbns | vectors_isbns)
    
    # ISBNs missing from AT LEAST ONE of the files
    missing_from_any = user_isbns - (subjects_isbns & vectors_isbns)

    missing_from_subjects = user_isbns - subjects_isbns
    missing_from_vectors = user_isbns - vectors_isbns

    # 5. Verification Logic
    if missing_from_subjects == missing_from_vectors:
        print(f"\nSUCCESS: Both sets are identical ({len(missing_from_subjects)} ISBNs).")
        
        # Write to TXT file
        with open("processed_data/missing_isbns.txt", 'w', encoding='utf-8') as out_file:
            for isbn in sorted(list(missing_from_subjects)):
                out_file.write(f"{isbn}\n")
        
        print(f"Results saved to missing_isbns.txt")
    else:
        print("\nWARNING: The sets are not identical despite having the same count.")
        # Optional: find the difference between the two missing sets
        diff = missing_from_subjects ^ missing_from_vectors
        print(f"There are {len(diff)} ISBNs that are missing in one file but present in the other.")

find_missing_isbns("processed_data/curated_users.jsonl", "processed_data/books_with_subjects.jsonl", "processed_data/book_vectors_base.jsonl")