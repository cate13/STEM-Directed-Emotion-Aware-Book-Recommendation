import csv
import json

# Step 1: Read JSONL and build an ISBN lookup dictionary
isbn_to_genre = {}
with open("STEM_Annontate/200_classified.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            raw_isbn = str(data.get("ISBN", "")).strip()
            genre = str(data.get("Genre", "")).strip()

            # Format 'Not' -> 'Not STEM', keeping 'STEM' as is
            formatted_genre = "Not STEM" if genre == "Not" else genre
            isbn_to_genre[raw_isbn] = formatted_genre

updated_rows = []
with open("STEM_Annontate/annontated_books.csv", "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames

    for row in reader:
        csv_isbn = str(row["isbn"]).strip()
        # Assign matched genre if ISBN exists in the JSONL lookup
        if csv_isbn in isbn_to_genre:
            row["user_3"] = isbn_to_genre[csv_isbn]
        updated_rows.append(row)

with open("STEM_Annontate/annontated_books_3.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(updated_rows)

print("Finished updating CSV!")