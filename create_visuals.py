import json
from tqdm import tqdm

from Recomender_Helper.vector_helper import combine_using_bilinear_pool_with_label, concat_with_labels, get_vector_by_isbn, saveGraphVector, average_vectors


with open("user_eval_sets/users_12_25_1_plus_STEM_books_and_10_plus_high_rated_split_60_40.json", 'r') as file:
    data = json.load(file)
    
    for item in tqdm(data):
        profile_books = item['candidate_profile']
        user_id = item['user_id']
        old_vectors = []
        new_vectors = []
        for book in profile_books:
            t_vec = get_vector_by_isbn(book['isbn'], "empath")
            e_vec = get_vector_by_isbn(book['isbn'], "emotion_intensity")
            old_pool = concat_with_labels(e_vec, t_vec)
            new_pool = combine_using_bilinear_pool_with_label(e_vec, t_vec)
            old_vectors.append(old_pool)
            new_vectors.append(new_pool)
        old_profile_vector = average_vectors(old_vectors)
        new_proflie_vector = average_vectors(new_vectors)
        saveGraphVector(old_profile_vector, f"{user_id}_concat", "visuals")
        saveGraphVector(new_proflie_vector, f"{user_id}_bilinear_pool", "visuals")