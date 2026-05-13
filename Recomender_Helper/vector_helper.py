import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Union
import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BOOK_VECTORS_BASE = os.path.join(
    BASE_DIR, "processed_data", "book_vectors_base.jsonl"
)

BOOK_VECTORS_7D = os.path.join(
    BASE_DIR, "processed_data", "book_vectors_for_empath_7D.jsonl"
)

def _load_book_data(file_path):
    book_map = {}
    print("loading vec data")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                isbn = record.get("isbn")
                if isbn:
                    # Store the whole record (or just the vectors) keyed by ISBN
                    book_map[isbn] = record
                    #print(isbn)
    except FileNotFoundError:
        print(f"Warning: {file_path} not found.")
    return book_map


# Global constant loaded when the script starts
BASE_BOOK_DATA_CACHE = _load_book_data(BOOK_VECTORS_BASE)
EMPATH_7D_DATA_CACHE = _load_book_data(BOOK_VECTORS_7D)

def get_vector_by_isbn(isbn: str, vector_type: str):
    valid_types = {"emotion_intensity", "emotion", "empath", "tf_idf", "empath_vec_with_base_word_list", "empath_vec_shared_llm_word_lsit", "empath_vec_chat_gpt_word_list", "empath_vec_gemini_word_list"}

    if vector_type in {"emotion_intensity", "emotion", "empath", "tf_idf"}:
        record = BASE_BOOK_DATA_CACHE.get(isbn)
    elif vector_type in {"empath_vec_with_base_word_list", "empath_vec_shared_llm_word_lsit", "empath_vec_chat_gpt_word_list", "empath_vec_gemini_word_list"}:
        record = EMPATH_7D_DATA_CACHE.get(isbn)
    else: raise ValueError(f"{vector_type} is invalid vector_type. Must be one of {valid_types}")
    
    if not record:
        # Better to return None or raise a specific KeyError if ISBN isn't found
        raise KeyError(f"ISBN {isbn} not found in database.")

    # 2. Get the specific vector from that record
    vector = record.get(vector_type)
    
    if vector is None:
        raise ValueError(f"Vector type '{vector_type}' missing for ISBN: {isbn}")

    return vector

def graphTF_IDF(results, title):
    plt.figure()
    plt.bar(range(len(results)), results)
    plt.xlabel("Term Index")
    plt.ylabel("TF-IDF Weight")
    plt.title(title)
    plt.show()

def graphDictVector(results, title):
    r = dict(results)
        
    plt.barh(range(len(r)), list(r.values()), align='center')
    plt.yticks(range(len(r)), list(r.keys()))

    #plt.xlabel('Emotion')
    plt.title(title)
    plt.show()

def graphVector(results, title):
    if isinstance(results, list):
        graphTF_IDF(results, title)
    else:
        graphDictVector(results, title)

def concat(vec1, vec2):
    # print(type(vec1))
    # print(type(vec2))
    if isinstance(vec1, dict):
        vec1 = list(vec1.values())
    if isinstance(vec2, dict):
        vec2 = list(vec2.values())
    return vec1 + vec2

def concat_with_weight(e_vec, t_vec, e_weight, t_weight):
    if isinstance(e_vec, dict):
        e_vec = list(e_vec.values())
    if isinstance(t_vec, dict):
        t_vec = list(t_vec.values())

    weighted_e = [x * e_weight for x in e_vec]
    weighted_t = [x * t_weight for x in t_vec]
    
    return weighted_e + weighted_t

def concat_with_weight_for_multi_topic_vec(e_vec, t_vecs, e_weight = 1.0, t_weight = 1.0):
    if isinstance(e_vec, dict):
        e_vec = list(e_vec.values())
    for i in range(len(t_vecs)):
        if isinstance(t_vecs[i], dict):
            t_vecs[i] = list(t_vecs[i].values())
    
    resulting_vec = [x * e_weight for x in e_vec]
    for t in t_vecs:
        resulting_vec = resulting_vec + [x * t_weight for x in t]
    return resulting_vec

def cosine_similarity(vec1, vec2):
    if type(vec1) is not type(vec2):
        raise ValueError("Both vectors must be the same type.")

    # Case 1: List vectors (e.g., tf_idf)
    if isinstance(vec1, list):
        if len(vec1) != len(vec2):
            raise ValueError("Both list vectors must have same length.")

        v1 = np.array(vec1, dtype=float)
        v2 = np.array(vec2, dtype=float)

    # Case 2: Dict vectors (e.g., emotion/empath)
    elif isinstance(vec1, dict):
        if set(vec1.keys()) != set(vec2.keys()):
            raise ValueError("Both dict vectors must have same keys.")

        # Ensure consistent ordering
        keys = sorted(vec1.keys())
        v1 = np.array([vec1[k] for k in keys], dtype=float)
        v2 = np.array([vec2[k] for k in keys], dtype=float)
    elif isinstance(vec1, np.ndarray):
        if vec1.shape != vec2.shape:
            raise ValueError("Both numpy vectors must have the same shape.")
        v1 = vec1
        v2 = vec2
    else:
        raise TypeError(f"Unsupported vector type. {vec1} {vec2}")

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(v1, v2) / (norm1 * norm2))

def average_vectors(vectors: List[Union[List[float], dict]]):
    """
    Averages a list of vectors.
    
    Works for:
        - List[float]  (e.g., tf_idf)
        - Dict[str, float] (e.g., emotion_intensity, emotion, empath)

    Returns:
        Averaged vector (same structure as input)
    """

    if not vectors:
        raise ValueError("Vector list is empty.")
    
    vectors = [v for v in vectors if v is not None]

    first = vectors[0]

    # Case 1: List-based vector (e.g., tf_idf)
    if isinstance(first, list):
        return np.mean(np.array(vectors), axis=0).tolist()

    # Case 2: NumPy-based vector
    elif isinstance(first, np.ndarray):
        # We average along the rows (axis 0)
        return np.mean(np.array(vectors), axis=0)
    
    # Case 3: Dict-based vector (emotion/empath)
    elif isinstance(first, dict):
        keys = first.keys()

        # Safety check
        if not all(v.keys() == keys for v in vectors):
            raise ValueError("All dict vectors must have same keys.")

        return {
            key: sum(v[key] for v in vectors) / len(vectors)
            for key in keys
        }

    else:
        raise TypeError(f"Unsupported vector type. {vectors}")

if __name__ == "__main__":
    isbn_1 = "0195153448"
    isbn_2 = "0399135782"
    isbn_3 = "0425176428"
    isbn_list = [isbn_1, isbn_2, isbn_3]
    for i in isbn_list:
        temp_t_1 = get_vector_by_isbn(i, "empath_vec_gemini_word_list")
        temp_t_2 = get_vector_by_isbn(i, "empath")
        temp_t_3 = get_vector_by_isbn(i, "empath_vec_shared_llm_word_lsit")
        temp_e = get_vector_by_isbn(i, "emotion")
        print(concat_with_weight_for_multi_topic_vec(temp_e, [temp_t_1, temp_t_2, temp_t_3], 0.2, 0.9))
        

