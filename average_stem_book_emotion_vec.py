from Recomender_Helper.vector_helper import get_vector_by_isbn, average_vectors, graphVector
import json

def load_stem_isbns(stem_path):
    """Load all STEM ISBNs into a set for fast lookup."""
    stem_isbns = set()

    with open(stem_path, "r", encoding="utf-8") as f:
        for line in f:
            book = json.loads(line)
            stem_isbns.add(book["ISBN"])

    return stem_isbns

stem_book_file_path = "processed_data/stem_books.jsonl"

stem_isbns = load_stem_isbns(stem_book_file_path)

emotion_vec = []
emotion_intensity_vec = []

for i in stem_isbns:
    emotion_vec.append(get_vector_by_isbn(i, "emotion"))
    emotion_intensity_vec.append(get_vector_by_isbn(i, "emotion_intensity"))

print(type(emotion_vec[0]))
print(type(emotion_intensity_vec[0]))

average_emotion_vec = average_vectors(emotion_vec)
average_emotion_intensity_vec = average_vectors(emotion_intensity_vec)

graphVector(average_emotion_vec, "Average Emotion Vector for STEM Books")
graphVector(average_emotion_intensity_vec, "Average Emotion Intensity Vector for STEM Books")
