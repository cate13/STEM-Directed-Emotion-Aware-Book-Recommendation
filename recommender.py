import json
import os
from tqdm import tqdm
from collections import Counter
from Recomender_Helper.vector_helper import get_vector_by_isbn, cosine_similarity, average_vectors, concat, concat_with_weight
from Vectorizer.EmotionConditionedTopicVectorMaker import EmotionConditionedTopicVectorMaker
from Vectorizer.CorrelationCombo import combine_vectors
import itertools
import numpy as np

# ---- Paths ----

TEST_DATA_FILE = "user_eval_sets/users_1_plus_STEM_books_and_6_plus_high_rated_split_60_40.json"
#TEST_DATA_FILE = "user_eval_sets/test.json"
STEM_BOOKS_FILE = "processed_data/stem_isbns_from_topic.txt"

def handle_book(isbn, emotion_type, topic_type, emotion_weight = 1.0, topic_weight = 1.0, use_matrix_combo = False, reduce = False):
    emotion_vec = get_vector_by_isbn(isbn, emotion_type)
    if emotion_vec is None:
        raise Exception(f"Do not have vector for {isbn}")
    topic_vec = get_vector_by_isbn(isbn, topic_type)

    if use_matrix_combo:
        return combine_vectors(emotion_vec, emotion_type, topic_vec, topic_type, reduce, emotion_weight)
    else:
        return concat_with_weight(emotion_vec, topic_vec, emotion_weight, topic_weight)

def general_stem_topic_vec_maker(topic_type):
    stem_isbns = set()
    print("Retreiving General STEM isbns")
    with open(STEM_BOOKS_FILE, 'r') as file:
        for line in tqdm(file):
            stem_isbns.add(line.strip())
    stem_vecs = []
    print("Retrieving STEM vectors")
    missing_stem_isbn = 0
    for isbn in tqdm(stem_isbns):
        try:
            temp = get_vector_by_isbn(isbn, topic_type)
            if temp:
                stem_vecs.append(temp)
        except:
            missing_stem_isbn += 1
    #print(f"no vector for {missing_stem_isbn} STEM books")
    return average_vectors(stem_vecs)

def make_candidate_profile(profile_books, emotion_type, general_stem_topic_vec, topic_type, emotion_weight = 1.0, topic_weight = 1.0, use_matrix_combo = False, reduce = False):
    emotion_vectors = []
    for book in profile_books:
        isbn = book['isbn']
        try:
            emotion_vectors.append(get_vector_by_isbn(isbn, emotion_type))
        except Exception as e:
            print(e)
    if len(emotion_vectors) == 0:
        raise Exception(f"Missing all profile books")
    
    emotion_profile = average_vectors(emotion_vectors)

    if use_matrix_combo:
        return combine_vectors(emotion_profile, emotion_type, general_stem_topic_vec, topic_type, reduce, emotion_weight)
    else:
        return concat_with_weight(emotion_profile, general_stem_topic_vec, emotion_weight, topic_weight)



def recommend(test_data_file, emotion_type = "emotion_intensity", topic_type = "tf_idf", emotion_weight = 1.0, topic_weight = 1.0, use_matrix_combo = False, reduce = False):
    general_stem_topic_vec = general_stem_topic_vec_maker(topic_type)
    with open(test_data_file, 'r') as file:
        data = json.load(file)
    
    for item in tqdm(data):
        profile_books = item['candidate_profile']
        try:
            candidate_profile = make_candidate_profile(profile_books, emotion_type, general_stem_topic_vec, topic_type, emotion_weight, topic_weight, use_matrix_combo, reduce)
            recomendation_books = item['recommendation_list']
            for book in recomendation_books:
                try:
                    book_vec = handle_book(book['isbn'], emotion_type, topic_type, emotion_weight, topic_weight, use_matrix_combo, reduce)
                    #book_vec = get_vector_by_isbn(book['isbn'], emotion_type)
                    cos = cosine_similarity(candidate_profile, book_vec)
                    book['cos'] = cos
                except Exception as e:
                    print(e)
        except Exception as e:
            print(e)
            print(item['user_id'])
    
    if use_matrix_combo & reduce:
        output_file_name = f"recommendations/12-25/{emotion_type}_reduced_{topic_type}_correlation_combo_with_{emotion_weight}.json"
    elif use_matrix_combo:
        output_file_name = f"recommendations/12-25/{emotion_type}_{topic_type}_correlation_combo_with_{emotion_weight}.json"
    else:
        output_file_name = f"recommendations/12-25/{emotion_type}_with_weight_{emotion_weight}_{topic_type}_with_weight_{topic_weight}.json"
    with open(output_file_name, 'w') as f:
        json.dump(data, f, indent=4)


def test_weights(emotion_type="emotion", topic_type="tf_idf"):
    for i in range(10):
        e_weight = (i + 1) / 10
        t_weight = 1.0 - (i / 10)

        recommend(TEST_DATA_FILE, emotion_type, topic_type, emotion_weight=e_weight, topic_weight=t_weight)

def run_all_recommendations(test_data_file):
    # Define the parameter ranges
    emotion_types = ["emotion_intensity", "emotion"]
    topic_types = ["empath", "tf_idf"]
    
    # Using np.linspace or np.arange to handle the 0.1 steps
    # We round to 1 decimal place to avoid floating point errors (e.g., 0.30000000000000004)
    weights = [round(x, 1) for x in np.arange(0.0, 1.1, 0.1)]
    
    matrix_combos = [True, False]
    
    # Generate all combinations
    all_combinations = itertools.product(
        emotion_types, 
        topic_types, 
        weights, 
        weights, 
        matrix_combos
    )

    for e_type, t_type, e_weight, t_weight, m_combo in all_combinations:
        
        # Constraint: reduce can only be True if topic_type is "empath"
        # Otherwise, we only run it as False
        reduce_options = [True, False] if t_type == "empath" else [False]
        
        for r_val in reduce_options:
            print(f"Running: {e_type}, {t_type}, e_w: {e_weight}, t_w: {t_weight}, m: {m_combo}, r: {r_val}")
            
            # Call your method
            recommend(
                test_data_file,
                emotion_type=e_type,
                topic_type=t_type,
                emotion_weight=e_weight,
                topic_weight=t_weight,
                use_matrix_combo=m_combo,
                reduce=r_val
            )


#recommend(TEST_DATA_FILE, emotion_type="emotion", topic_type="empath", emotion_weight = 0.001, topic_weight = 1.0)

#test_weights(topic_type="sentance_bert")

#run_all_recommendations(test_data_file=TEST_DATA_FILE)

recommend(TEST_DATA_FILE, emotion_type="emotion", topic_type="empath", use_matrix_combo=True)