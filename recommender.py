import json
import os
from tqdm import tqdm
from collections import Counter
from Recomender_Helper.vector_helper import get_vector_by_isbn, cosine_similarity, average_vectors, combine_using_bilinear_pool, concat_with_weight, concat_with_weight_for_multi_topic_vec
from Vectorizer.EmotionConditionedTopicVectorMaker import EmotionConditionedTopicVectorMaker
from Vectorizer.CorrelationCombo import combine_vectors
import itertools
import numpy as np
from itertools import combinations

# ---- Paths ----

TEST_DATA_FILE = "user_eval_sets/users_12_25_1_plus_STEM_books_and_10_plus_high_rated_split_60_40.json"
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

def handle_book_for_multi_topic(isbn, emotion_type, topic_type_list, emotion_weight = 1.0, topic_weight = 1.0):
    emotion_vec = get_vector_by_isbn(isbn, emotion_type)
    topic_vec_list = []
    for t in topic_type_list:
        topic_vec_list.append(get_vector_by_isbn(isbn, t))
    return concat_with_weight_for_multi_topic_vec(emotion_vec, topic_vec_list, emotion_weight, topic_weight)

def handle_book_for_bilinear_pool(isbn, emotion_type, topic_type):
    emotion_vec = get_vector_by_isbn(isbn, emotion_type)
    topic_vec = get_vector_by_isbn(isbn, topic_type)
    return combine_using_bilinear_pool(emotion_vec, topic_vec)


def general_stem_topic_vec_maker_multi_topic(topic_type_list):
    stem_isbns = set()
    print("Retreiving General STEM isbns")
    with open(STEM_BOOKS_FILE, 'r') as file:
        for line in tqdm(file):
            stem_isbns.add(line.strip())
    stem_vecs = []

    for isbn in tqdm(stem_isbns):
        whole_topic_list = []
        for t in topic_type_list:
            temp = get_vector_by_isbn(isbn, t)
            if isinstance(temp, dict):
                temp = list(temp.values())
            whole_topic_list = whole_topic_list + temp
        if whole_topic_list:
            stem_vecs.append(whole_topic_list)
    return average_vectors(stem_vecs)
    

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

def make_candidate_profile_bilinear_pool(profile_books, emotion_type, general_stem_topic_vec):
    emotion_vectors = []
    # print(profile_books)
    for book in profile_books:
        isbn = book['isbn']
        try:
            emotion_vectors.append(get_vector_by_isbn(isbn, emotion_type))
        except Exception as e:
            print(e)
    if len(emotion_vectors) == 0:
        raise Exception(f"Missing all profile books")
    emotion_profile = average_vectors(emotion_vectors)
    return combine_using_bilinear_pool(emotion_profile, general_stem_topic_vec)

def make_candidate_profile_for_multi_topic(profile_books, emotion_type, general_stem_topic_vec, emotion_weight = 1.0, topic_weight = 1.0):
    # since the general stem vector has already had all the types added 
    # concat_with_weight_for_multi_topic_vec isn't needed 
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
    return concat_with_weight(emotion_profile, general_stem_topic_vec, emotion_weight, topic_weight)

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

def recommend_bilinear_pool(test_data_file, output_folder, emotion_type = "emotion_intensity", topic_type = "empath"):
    general_stem_topic_vec = general_stem_topic_vec_maker(topic_type)
    with open(test_data_file, 'r') as file:
        data = json.load(file)
    
    for item in tqdm(data):
        profile_books = item['candidate_profile']
        candidate_profile = make_candidate_profile_bilinear_pool(profile_books, emotion_type, general_stem_topic_vec)
        recomendation_books = item['recommendation_list']
        for book in recomendation_books:
            book_vec = handle_book_for_bilinear_pool(book['isbn'], emotion_type, topic_type)
            cos = cosine_similarity(candidate_profile, book_vec)
            book['cos'] = cos
    
    output_file_path = f"{output_folder}/bilinear_pool_{emotion_type}_{topic_type}.json"
    with open(output_file_path, 'w') as f:
        json.dump(data, f, indent=4)

def recommend_multi_topic(test_data_file, output_folder, emotion_type = "emotion_intensity", topic_types = ["tf_idf"], emotion_weight = 1.0, topic_weight = 1.0):
    print(topic_types)
    general_stem_topic_vec = general_stem_topic_vec_maker_multi_topic(topic_types) 
    with open(test_data_file, 'r') as file:
        data = json.load(file)

    for item in tqdm(data):
        profile_books = item['candidate_profile']
        try: 
            candidate_profile = make_candidate_profile_for_multi_topic(profile_books, emotion_type, general_stem_topic_vec, emotion_weight, topic_weight)
            recomendation_books = item['recommendation_list']
            for book in recomendation_books:
                book_vec = handle_book_for_multi_topic(book['isbn'], emotion_type, topic_types, emotion_weight, topic_weight)
                cos = cosine_similarity(candidate_profile, book_vec)
                book['cos'] = cos
        except Exception as e:
            print(e)
            print(item['user_id'])
    topic_string = "_".join(topic_types)
    output_file_path = f"{output_folder}/{emotion_type}_with_weight_{emotion_weight}_{topic_string}_with_weight_{topic_weight}.json"
    with open(output_file_path, 'w') as f:
        json.dump(data, f, indent=4)
    

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
        output_file_name = f"recommendations/12-25_age_10_plus_highly_rated_books/empath_7D/{emotion_type}_reduced_{topic_type}_correlation_combo_with_{emotion_weight}.json"
    elif use_matrix_combo:
        output_file_name = f"recommendations/12-25_age_10_plus_highly_rated_books/empath_7D/{emotion_type}_{topic_type}_correlation_combo_with_{emotion_weight}.json"
    else:
        output_file_name = f"recommendations/12-25_age_10_plus_highly_rated_books/empath_7D/{emotion_type}_with_weight_{emotion_weight}_{topic_type}_with_weight_{topic_weight}.json"
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
    matrix_options = [True, False]
    
    # Generate the 11 weight pairs (0.0/1.0, 0.1/0.9 ... 1.0/0.0)
    weights = [(round(w, 1), round(1.0 - w, 1)) for w in np.arange(0.0, 1.1, 0.1)]

    all_configs = []

    for e_type in emotion_types:
        for t_type in topic_types:
            for e_w, t_w in weights:
                for use_matrix in matrix_options:
                    
                    # Logic for the 'reduce' flag
                    reduce_options = [False]
                    if t_type == "empath":
                        reduce_options = [True, False]
                    
                    for should_reduce in reduce_options:
                        all_configs.append({
                            "emotion_type": e_type,
                            "topic_type": t_type,
                            "emotion_weight": e_w,
                            "topic_weight": t_w,
                            "use_matrix_combo": use_matrix,
                            "reduce": should_reduce
                        })

    for c in all_configs:
        recommend(
            test_data_file, 
            emotion_type=c["emotion_type"],
            topic_type=c["topic_type"],
            emotion_weight=c["emotion_weight"],
            topic_weight=c["topic_weight"],
            use_matrix_combo=c["use_matrix_combo"],
            reduce=c["reduce"]
        )

def run_emotion_v_empath(test_data_file):
    emotion_types = ["emotion_intensity", "emotion"]
    topic_type = "empath"

    emotion_weights = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    topic_weights = [1.0, 1.5, 2.0, 7.5, 10.0, 20.0]

    combinations = itertools.product(emotion_types, emotion_weights, topic_weights)

    for e_type, e_weight, t_weight in tqdm(combinations):
        recommend(
            test_data_file, 
            emotion_type=e_type,
            topic_type=topic_type,
            emotion_weight=e_weight,
            topic_weight=t_weight,
            use_matrix_combo=False,
            reduce=False,
        )

def run_empath_7D(test_data_file):
    emotion_types = ["emotion_intensity", "emotion"]
    topic_types = ["empath_vec_with_base_word_list", "empath_vec_shared_llm_word_lsit", "empath_vec_chat_gpt_word_list", "empath_vec_gemini_word_list"]

    for t_type in topic_types:
        for e_type in tqdm(emotion_types):
            recommend(test_data_file, emotion_type=e_type, topic_type=t_type)
            recommend(test_data_file, emotion_type=e_type, topic_type=t_type, emotion_weight=0.1, topic_weight=0.9)

def run_bilinear_pool(test_data_file):
    emotion_types = ["emotion_intensity", "emotion"]
    topic_types = ["empath", "tf_idf", "empath_vec_shared_llm_word_lsit"]

    for e in emotion_types:
        for t in topic_types:
            recommend_bilinear_pool(test_data_file, "recommendations/12-25_age_10_plus_highly_rated_books/bilinear_pool", e, t)


def run_multi_topic_vec_rec():
    topic_types = ["empath", "tf_idf", "empath_vec_with_base_word_list", "empath_vec_shared_llm_word_lsit", "empath_vec_chat_gpt_word_list", "empath_vec_gemini_word_list"]

    all_combinations = []

    for r in range(1, len(topic_types) + 1):
        all_combinations.extend(combinations(topic_types, r))

    # optional: convert tuples to lists
    all_combinations = [list(c) for c in all_combinations]

    for topic_combo in tqdm(all_combinations):
        recommend_multi_topic(test_data_file=TEST_DATA_FILE, output_folder="recommendations/12-25_age_10_plus_highly_rated_books/multi_topic_test", emotion_type="emotion", topic_types=topic_combo)
        recommend_multi_topic(test_data_file=TEST_DATA_FILE, output_folder="recommendations/12-25_age_10_plus_highly_rated_books/multi_topic_test", emotion_type="emotion", topic_types=topic_combo, topic_weight=0.9, emotion_weight=0.1)
        recommend_multi_topic(test_data_file=TEST_DATA_FILE, output_folder="recommendations/12-25_age_10_plus_highly_rated_books/multi_topic_test", emotion_type="emotion_intensity", topic_types=topic_combo)
        recommend_multi_topic(test_data_file=TEST_DATA_FILE, output_folder="recommendations/12-25_age_10_plus_highly_rated_books/multi_topic_test", emotion_type="emotion_intensity", topic_types=topic_combo, topic_weight=0.9, emotion_weight=0.1)

#recommend(TEST_DATA_FILE, emotion_type="emotion", topic_type="empath", emotion_weight = 0.001, topic_weight = 1.0)

#test_weights(topic_type="sentance_bert")

run_bilinear_pool(TEST_DATA_FILE)

# recommend_multi_topic(test_data_file=TEST_DATA_FILE, output_folder="recommendations/12-25_age_10_plus_highly_rated_books/multi_topic_test", emotion_type="emotion", topic_types=["empath", "empath_vec_with_base_word_list"])

#recommend(TEST_DATA_FILE, emotion_type="emotion", topic_type="empath", reduce=True)