import re

def is_valid_isbn10(isbn):
    if not re.fullmatch(r"\d{9}[\dX]", isbn):
        return False
    total = 0
    for i, char in enumerate(isbn[:9]):
        total += (i + 1) * int(char)
    check = total % 11
    return str(check) == isbn[-1] or (check == 10 and isbn[-1] == "X")

def is_valid_isbn13(isbn):
    if not re.fullmatch(r"\d{13}", isbn):
        return False
    total = 0
    for i, char in enumerate(isbn[:12]):
        factor = 1 if i % 2 == 0 else 3
        total += int(char) * factor
    check = (10 - (total % 10)) % 10
    return str(check) == isbn[-1]

def clean_isbn(line):
    return re.sub(r"[^\dX]", "", line.upper())

valid = []
invalid = []

with open("books_read_by_youth_missing.txt", "r") as f:
    for line in f:
        raw = line.strip()
        cleaned = clean_isbn(raw)

        if is_valid_isbn10(cleaned) or is_valid_isbn13(cleaned):
            valid.append(raw)
        else:
            invalid.append(raw)

with open("valid_isbns.txt", "w") as f:
    f.write("\n".join(valid))

with open("invalid_isbns.txt", "w") as f:
    f.write("\n".join(invalid))

print("Done.")