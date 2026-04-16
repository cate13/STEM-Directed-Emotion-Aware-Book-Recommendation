import json
from pathlib import Path


def load_txt_isbns(txt_path):
    """Load ISBNs from a plain text file (one per line)."""
    with open(txt_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def load_jsonl_isbns(jsonl_path):
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

    txt_path = base_dir / "isbn_files" / "books_read_by_youth.txt"
    jsonl_path = base_dir.parent / "processed_data" / "books_with_subjects_combined.jsonl"

    txt_isbns = load_txt_isbns(txt_path)
    jsonl_isbns = load_jsonl_isbns(jsonl_path)

    missing_isbns = sorted(txt_isbns - jsonl_isbns)
    file_path = base_dir / "isbn_files" / "books_read_by_youth_missing.txt"

    with open(file_path, 'w') as f:
        for isbn in missing_isbns:
            f.write(str(isbn) + "\n")

if __name__ == "__main__":
    main()