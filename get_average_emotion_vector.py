
import json
from Recomender_Helper.vector_helper import get_vector_by_isbn, graphVector, average_vectors

JSONL_PATH = "processed_data/books_with_subjects_combined.jsonl"

all_book_vectors = []

with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        book = json.loads(line)
        isbn = book.get("ISBN")
        all_book_vectors.append(get_vector_by_isbn(isbn, "emotion"))

average_emotion = average_vectors(all_book_vectors)
print(f"Average Emotion Vector: {average_emotion}")

graphVector(average_emotion, "Average Emotion Vector All Books")
