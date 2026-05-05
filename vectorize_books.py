import json
from Vectorizer.EmotionVectorMaker import EmotionVectorMaker
from Vectorizer.EmpathVectorMaker import EmpathVectorMaker
from Vectorizer.TF_IDFVectorMaker import TF_IDFVectorMaker
from Vectorizer.Empath7DVectorMaker import Empath7DVectorMaker
from Vectorizer.TF_IDF_long_VectorMaker import TF_IDF_long_VectorMaker
from Vectorizer.SentanceBERTVectorMaker import SBERTVectorMaker
from Vectorizer.SentanceBERTshortVectorMaker import SBERTshortVectorMaker
import os
from tqdm import tqdm

def get_descriptions(JSONL_PATH):

    descriptions = []

    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            book = json.loads(line)
            isbn = book.get("ISBN")

            description = book.get("description")
            if description:  # avoid None or empty
                descriptions.append((isbn, description))
    
    return descriptions

def get_stem_isbns():
    stem_books = set()
    with open("processed_data/stem_isbns_from_topic.txt", encoding="utf-8") as f:
        for line in f:
            stem_books.add(line.strip())
    with open("processed_data/stem_isbns_from_cosine.txt", encoding="utf-8") as f:
        for line in f:
            stem_books.add(line.strip())
    return stem_books

def vectorize_non_stem_books_for_classification():
    JSONL_PATH = "processed_data/books_with_subjects.jsonl"

    book_descriptions = get_descriptions(JSONL_PATH)

    empath_vec_maker = Empath7DVectorMaker()
    sbert_vec_maker = SBERTVectorMaker()

    with open("Classifier/books_for_stem_classification.jsonl", "w", encoding="utf-8") as outfile:
        for isbn, description in tqdm(book_descriptions):
            empath_vec = empath_vec_maker.getEmapthVector(description)
            sbert_vec = sbert_vec_maker.get_vector(description)

            book = {
                "isbn" : isbn,
                "empath_7D" : empath_vec,
                "SBERT" : sbert_vec.tolist()
            }
            outfile.write(json.dumps(book) + "\n")

def updated_vectorizer():
    JSONL_PATH = "processed_data/books_with_subjects.jsonl"
    book_descriptions = get_descriptions(JSONL_PATH)
    print("Loaded Book Descriptions")

    empath_7D = Empath7DVectorMaker()
    s_bert = SBERTVectorMaker()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(
        BASE_DIR, "processed_data", "book_vectors_extended.jsonl"
    )

    with open(output_path, "w", encoding="utf-8") as outfile:
        i = 0
        for isbn, description in tqdm(book_descriptions):
            e7 = empath_7D.getEmapthVector(description)
            sbert = s_bert.get_vector(description)
            book = {
                "isbn" : isbn,
                "empath_7D" : e7,
                "sentance_bert" : sbert.tolist()
            }
            outfile.write(json.dumps(book) + "\n")



def redo_vectorizer():
    younger_books = "processed_data/books_with_subjects_read_by_younger_readers.jsonl"
    older_books = "processed_data/books_with_subjects_read_by_older_readers.jsonl"
    young_book_descriptions = get_descriptions(younger_books)
    print(young_book_descriptions[0])
    print(f"Loaded {len(young_book_descriptions)} books from by younger readers, used to create TF_IDF maker")
    old_book_descriptions = get_descriptions(older_books)
    print(old_book_descriptions[0])
    print(f"Loaded {len(old_book_descriptions)} books from by older readers")
    print("Loaded Book Descriptions")

    prev_descriptions = [text for _, text in young_book_descriptions]

    tf_idf_vec_maker = TF_IDFVectorMaker(prev_descriptions)

    emo_intensity_vec_maker = EmotionVectorMaker()

    emo_vec_maker = EmotionVectorMaker(use_intensity=False)

    empath_vec_maker = EmpathVectorMaker()

    print("++++++++++ALL VECTOR MAKERS MADE++++++++++")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(
        BASE_DIR, "processed_data", "book_vectors_older_readers.jsonl"
    )

    with open(output_path, "w", encoding="utf-8") as outfile:
        i = 0
        for isbn, description in tqdm(old_book_descriptions):
            emo_intensity_vec = emo_intensity_vec_maker.getEmotionVector(description, removeObj=True)
            emo_vec = emo_vec_maker.getEmotionVector(description, removeObj=True)
            empath_vec = empath_vec_maker.getEmapthVector(description)
            tf_idf_vec = tf_idf_vec_maker.getTF_IDFvector(description)
            book = {
                "isbn" : isbn,
                "emotion_intensity" : emo_intensity_vec,
                "emotion" : emo_vec,
                "empath" : empath_vec,
                "tf_idf" : tf_idf_vec.tolist()
            }
            outfile.write(json.dumps(book) + "\n")

def base_vectorizer():
    JSONL_PATH = "processed_data/books_with_subjects_complete.jsonl"

    book_descriptions = get_descriptions(JSONL_PATH)

    emo_intensity_vec_maker = EmotionVectorMaker()

    emo_vec_maker = EmotionVectorMaker(use_intensity=False)

    empath_vec_maker = EmpathVectorMaker()

    all_descriptions = [text for _, text in book_descriptions]

    tf_idf_vec_maker = TF_IDFVectorMaker(all_descriptions)

    print("Vector Makers all made")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(
        BASE_DIR, "processed_data", "book_vector_readers_read_stem.jsonl"
    )

    with open(output_path, "w", encoding="utf-8") as outfile:
        i = 0
        for isbn, description in tqdm(book_descriptions):
            emo_intensity_vec = emo_intensity_vec_maker.getEmotionVector(description, removeObj=True)
            emo_vec = emo_vec_maker.getEmotionVector(description, removeObj=True)
            empath_vec = empath_vec_maker.getEmapthVector(description)
            tf_idf_vec = tf_idf_vec_maker.getTF_IDFvector(description)
            book = {
                "isbn" : isbn,
                "emotion_intensity" : emo_intensity_vec,
                "emotion" : emo_vec,
                "empath" : empath_vec,
                "tf_idf" : tf_idf_vec.tolist()
            }
            outfile.write(json.dumps(book) + "\n")


def get_even_stem_not_stem_vectorized():
    JSONL_PATH = "processed_data/books_with_subjects_complete.jsonl"
    STEM_ISBN_PATH = "data_exploring/stem_books_read_by_youth.txt"
    NON_STEM_ISBN_PATH = "data_exploring/750_non_stem_books_read_by_youth_with_descriptions.txt"

    def get_descriptions():
        with open(STEM_ISBN_PATH, "r", encoding="utf-8") as f:
            set_1 = {line.strip() for line in f if line.strip()}

        # Load the second set
        with open(NON_STEM_ISBN_PATH, "r", encoding="utf-8") as f:
            set_2 = {line.strip() for line in f if line.strip()}

        # Combine them into one master set
        all_youth_isbns = set_1 | set_2

        descriptions = []

        with open(JSONL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                book = json.loads(line)
                isbn = book.get("ISBN")

                if isbn in all_youth_isbns:
                    description = book.get("description")
                    if description:  # avoid None or empty
                        descriptions.append((isbn, description))
        
        return descriptions

    book_descriptions = get_descriptions()

    empath_vec_maker = Empath7DVectorMaker()

    all_descriptions = [text for _, text in book_descriptions]

    tf_idf_vec_maker = TF_IDFVectorMaker(all_descriptions)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(
        BASE_DIR, "Classifier", "empath_7d_tf_idf_short_vectors.jsonl"
    )

    with open(output_path, "w", encoding="utf-8") as outfile:
        i = 0
        for isbn, description in tqdm(book_descriptions):
            empath_vec = empath_vec_maker.getEmapthVector(description)
            tf_idf = tf_idf_vec_maker.getTF_IDFvector(description)
            book = {
                "isbn" : isbn,
                "empath" : empath_vec,
                "tf_idf" : tf_idf.tolist()
            }
            outfile.write(json.dumps(book) + "\n")


def update_vectorizer_with_new_features():
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_PATH = os.path.join(BASE_DIR, "processed_data", "book_vectors_partial.jsonl")
    OUTPUT_PATH = os.path.join(BASE_DIR, "processed_data", "book_vectors.jsonl")
    JSONL_PATH = "processed_data/books_with_subjects_1.jsonl"
    
    # 1. Re-fetch descriptions to get the text for the new vectors
    # (Assuming get_descriptions() is accessible or copied here)
    print("Fetching descriptions...")
    book_descriptions = dict(get_descriptions(JSONL_PATH)) # Convert to dict for O(1) ISBN lookup

    # 2. Initialize your NEW VectorMakers here
    print("Initializing SBERT")
    sentence_bert_long = SBERTVectorMaker()
    #all_descriptions = list(book_descriptions.values())
    #print("Initializing SBERT short")
    #sentence_bert_short = SBERTshortVectorMaker(all_descriptions)
    print("Initializing Empath 7D")
    empath_7d = Empath7DVectorMaker()

    # 3. Process the existing file and add new fields
    print("Updating JSONL file...")
    with open(INPUT_PATH, "r", encoding="utf-8") as infile, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as outfile:
        
        for line in tqdm(infile):
            book = json.loads(line)
            isbn = book.get("isbn")
            
            # Get the original description for this ISBN
            description = book_descriptions.get(isbn)
            
            if description:
                book["sentanceBERT"] = sentence_bert_long.get_vector(description).tolist()
                #book["sentanceBert_short"] = sentence_bert_short.get_vector(description)
                book["empath_7d"] = empath_7d.getEmapthVector(description)
                
                # Write the updated dictionary back to the temp file
                outfile.write(json.dumps(book) + "\n")
            else:
                # If for some reason the description isn't found, 
                # keep the original data or skip
                outfile.write(json.dumps(book) + "\n")


redo_vectorizer()