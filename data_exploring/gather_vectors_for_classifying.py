import json
import os
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_VECTORS = os.path.join(
    BASE_DIR, "processed_data", "book_vectors_base.jsonl"
)

EXTRA_EMPATH_VECTORS = os.path.join(
    BASE_DIR, "processed_data", "book_vectors_for_empath_7D.jsonl"
)

STEM_ISBNS = os.path.join(
    BASE_DIR, "data_exploring", "isbn_files", "stem_books_read_by_youth.txt"
)

NON_STEM_ISBNS = os.path.join(
    BASE_DIR, "data_exploring", "isbn_files", "750_non_stem_books_read_by_youth_with_descriptions.txt"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR, "Classifier", "book_vectors_trainging_classifier.jsonl"
) 


def load_isbns(file_path):
    """Reads ISBNs from a text file, stripping whitespace."""
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return set()
    with open(file_path, 'r') as f:
        return {line.strip() for line in f if line.strip()}
    

def gather_vectors():
    stem_isbns = load_isbns(STEM_ISBNS)
    non_stem_isbns = load_isbns(NON_STEM_ISBNS)

    target_isbns = non_stem_isbns.union(stem_isbns)
    print(len(target_isbns))

    base_data = {}
    with open(BASE_VECTORS, 'r') as f:
        for line in f:
            data = json.loads(line)
            isbn = data.get("isbn")
            if isbn in target_isbns:
                base_data[isbn] = {
                    "tf_idf": data.get("tf_idf"),
                    "empath": data.get("empath")
                }
    
    empath_7d_data = {}
    with open(EXTRA_EMPATH_VECTORS, 'r') as f:
        for line in f:
            data = json.loads(line)
            isbn = data.get("isbn")
            if isbn in target_isbns:
                empath_7d_data[isbn] = {
                    "empath_vec_with_base_word_list": data.get("empath_vec_with_base_word_list"),
                    "empath_vec_shared_llm_word_lsit": data.get("empath_vec_shared_llm_word_lsit")
                }
    
    with open(OUTPUT_FILE, 'w') as out_f:
        for isbn in tqdm(target_isbns):
            # Only write if the ISBN exists in both data sources to avoid null vectors
            if isbn in base_data and isbn in empath_7d_data:
                combined_entry = {"isbn": isbn}
                combined_entry.update(base_data[isbn])
                combined_entry.update(empath_7d_data[isbn])
                
                out_f.write(json.dumps(combined_entry) + '\n')

gather_vectors()