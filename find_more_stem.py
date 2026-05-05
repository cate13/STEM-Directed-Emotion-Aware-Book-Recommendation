import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Union
from tqdm import tqdm
from Recomender_Helper.vector_helper import get_vector_by_isbn, average_vectors, cosine_similarity



def general_stem_topic_vec_maker(topic_type):
    stem_isbns = set()
    print("Retreiving General STEM isbns")
    with open("processed_data/stem_isbns_from_classifier.txt", 'r') as file:
        for line in file:
            stem_isbns.add(line.strip())
    print(len(stem_isbns))
    stem_vecs = []
    print("Retrieving STEM vectors")
    for isbn in tqdm(stem_isbns):
        print(isbn)
        temp = get_vector_by_isbn(isbn, topic_type)
        if temp:
            stem_vecs.append(temp)
    
    avg_vec = average_vectors(stem_vecs)
    return avg_vec, stem_isbns

def find_more_stem(avg_stem_vector, topic_type, threshold=0.75):
    candidate_books = []
    
    with open("processed_data/book_vectors_base_older_readers.jsonl", 'r') as file:
        for line in file:
            record = json.loads(line)
            isbn = record.get("isbn")
            stem_vec = get_vector_by_isbn(isbn, topic_type)

            similarity = cosine_similarity(stem_vec, avg_stem_vector)
        
            if similarity >= threshold:
                candidate_books.append(isbn)
    
    candidate_books.sort(key=lambda x: x[1], reverse=True)
    with open("processed_data/stem_isbns_from_cossine_new.txt", "w", encoding="utf-8") as f:
        for book in candidate_books:
            f.write(json.dumps(book) + "\n")

    return candidate_books


if __name__ == "__main__":
    print("Generating STEM topic vector using empath: ")
    avg_stem_vector, stem_isbns = general_stem_topic_vec_maker("empath")

    print("Searching for more STEM books . . .")
    # possible_books = find_more_stem(avg_stem_vector, "empath")
    # possible_books.sort(key=lambda x: x[1], reverse=True)
