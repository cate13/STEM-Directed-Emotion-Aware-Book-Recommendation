import json
from Vectorizer.Empath7DVectorMaker import Empath7DVectorMaker
from Recomender_Helper.vector_helper import graphVector, average_vectors, get_vector_by_isbn
from tqdm import tqdm


def get_isbns_with_vectors():
    path = "processed_data/book_vectors_attempt_1.jsonl"
    book_with_vec = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            book = json.loads(line)
            isbn = book.get("isbn")
            book_with_vec.add(isbn)
    return book_with_vec


def get_empath_7d_vectors(JSONL_PATH):
    only_use = get_isbns_with_vectors()

    vectors = {}
    EmpathVectorizer = Empath7DVectorMaker()

    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            book = json.loads(line)
            isbn = book.get("ISBN")
            if isbn not in only_use: continue

            description = book.get("description")
            if description:  # avoid None or empty
                vectors[isbn] = EmpathVectorizer.getEmapthVector(description)
    
    return vectors

def find_book_strong_domain():
    data = get_empath_7d_vectors("processed_data/books_with_subjects_combined.jsonl")
    print("Descritions Retrieved")

    categories = {key: [] for key in data[next(iter(data))].keys()}

    for isbn, vector in tqdm(data.items()):
        # Find the category with the highest value
        # key=vector.get tells max() to compare the values, not the strings
        top_category = max(vector, key=vector.get)
        
        # Check if the largest value is actually above 0
        if vector[top_category] > 0:
            categories[top_category].append(isbn)

    return categories


def save_to_file():
    data = find_book_strong_domain()
    with open("classified_isbns_limited_to_have_vec.txt", "w") as f:
        for category, isbns in data.items():
            # Write the header
            f.write(f"=== {category} ===\n")
            
            # Write each ISBN followed by a newline
            for isbn in isbns:
                f.write(f"{isbn}\n")
            
            # Add an extra newline between sections for readability
            f.write("\n")


def calculate_averages():
    categories = find_book_strong_domain()

    stem_domains = list(categories.keys())

    for cat in stem_domains:
        cat_list = []
        isbns = categories[cat]
        for i in tqdm(isbns):
            cat_list.append(get_vector_by_isbn(i, "tf_idf"))
        average_vec = average_vectors(cat_list)
        print(f"{cat} average vector: {average_vec}")
        graphVector(average_vec, f"{cat} average vector")

save_to_file()