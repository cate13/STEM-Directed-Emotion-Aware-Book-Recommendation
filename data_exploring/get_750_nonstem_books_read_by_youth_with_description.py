import json
from pathlib import Path
import random


def load_txt_isbns(txt_path):
    """Load ISBNs from a plain text file (one per line)."""
    with open(txt_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def load_books_with_description(jsonl_path):
    """Load ISBNs from a JSONL file (one JSON object per line)."""
    isbns = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                isbns.add(data["ISBN"].strip())
    return isbns

def main():
    base_dir = Path(__file__).resolve().parent

    stem_txt_path = base_dir.parent / "data_exploring" / "stem_books_read_by_youth.txt"
    txt_path = base_dir.parent / "data_exploring" / "books_read_by_youth.txt"
    
    description_books = base_dir.parent / "processed_data" / "books_with_subjects_complete.jsonl"

    stem_txt_isbns = load_txt_isbns(stem_txt_path)
    all_youth_isbns = load_txt_isbns(txt_path)
    jsonl_isbns = load_books_with_description(description_books)

    final_isbns = (all_youth_isbns - stem_txt_isbns) & jsonl_isbns
    sampled_isbns = random.sample(list(final_isbns), 750)
    file_path = base_dir.parent / "data_exploring" / "750_non_stem_books_read_by_youth_with_descriptions.txt"

    with open(file_path, 'w') as f:
        for isbn in sampled_isbns:
            f.write(str(isbn) + "\n")

if __name__ == "__main__":
    main()