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
    

def recommend(test_data_file, output_folder, emotion_type = "emotion_intensity", topic_type = "tf_idf", emotion_weight = 1.0, topic_weight = 1.0, matrix_combo = False, reduce = False):
    general_stem_topic_vec = general_stem_topic_vec_maker(topic_type)
    with open(test_data_file, 'r') as file:
        data = json.load(file)
    
    for item in tqdm(data):
        profile_books = item['candidate_profile']
        try:
            candidate_profile = make_candidate_profile(profile_books, emotion_type, general_stem_topic_vec, topic_type, emotion_weight, topic_weight, use_matrix_combo=matrix_combo, reduce=reduce)
            recomendation_books = item['recommendation_list']
            for book in recomendation_books:
                try:
                    book_vec = handle_book(book['isbn'], emotion_type, topic_type, emotion_weight, topic_weight, use_matrix_combo=matrix_combo, reduce=reduce)
                    #book_vec = get_vector_by_isbn(book['isbn'], emotion_type)
                    cos = cosine_similarity(candidate_profile, book_vec)
                    book['cos'] = cos
                except Exception as e:
                    print(e)
        except Exception as e:
            print(e)
            print(item['user_id'])
    
    if matrix_combo:
        output_file_name = f"{output_folder}/matrix_combo_{emotion_type}_{topic_type}_with_weight_{emotion_weight}.json"
    else:
        output_file_name = f"{output_folder}/{emotion_type}_with_weight_{emotion_weight}_{topic_type}_with_weight_{topic_weight}.json"
    with open(output_file_name, 'w') as f:
        json.dump(data, f, indent=4)


def test_weights(emotion_type="emotion", topic_type="tf_idf"):
    for i in range(10):
        e_weight = (i + 1) / 10
        t_weight = 1.0 - (i / 10)

        recommend(TEST_DATA_FILE, "recommendations/12-25_age_10_plus_highly_rated_books/sample_comparison", emotion_type=emotion_type, topic_type=topic_type, emotion_weight=e_weight, topic_weight=t_weight)

#recommend_bilinear_pool(TEST_DATA_FILE, "recommendations/12-25_age_10_plus_highly_rated_books/sample_comparison", emotion_type="emotion_intensity", topic_type="tf_idf")

def run_correlation_matrix_combo_reduce(test_data_file, output_folder):
    emotion_types = ["emotion_intensity", "emotion"]
    topic_types = ["empath"]

    for emotion_type in emotion_types:
        for topic_type in topic_types:
            for i in range(1, 10):
                emotion_weight = round(i * 0.1, 1)
                recommend(
                    test_data_file=test_data_file,
                    output_folder=output_folder,
                    emotion_type=emotion_type,
                    topic_type=topic_type,
                    emotion_weight=emotion_weight,
                    topic_weight=1.0,
                    matrix_combo=True,
                    reduce=True
                )


def run_base_weight_recommendation_combinations(test_data_file, output_folder):
    # Define the parameter spaces
    emotion_types = ["emotion_intensity", "emotion"]
    topic_types = ["tf_idf", "empath"]
    
    # Iterate through each combination of types
    for emotion_type in emotion_types:
        for topic_type in topic_types:
            
            # Loop for weights: 1 to 9 inclusive represents 0.1 to 0.9
            for i in range(1, 10):
                # Use rounding to prevent floating-point precision quirks
                emotion_weight = round(i * 0.1, 1)
                topic_weight = round(1.0 - emotion_weight, 1)
                
                # Optional: Print statement to track progress in the console
                print(f"Running: {emotion_type} | {topic_type} | "
                      f"Weights: {emotion_weight} / {topic_weight}")
                
                # Call your original recommend method
                recommend(
                    test_data_file=test_data_file,
                    output_folder=output_folder,
                    emotion_type=emotion_type,
                    topic_type=topic_type,
                    emotion_weight=emotion_weight,
                    topic_weight=topic_weight
                )

def run_base_combo(output_file = "recommendations/12-25_age_10_plus_highly_rated_books/OG_base_combo"):
    recommend(TEST_DATA_FILE, output_file, "emotion", "tf_idf", 1.0, 1.0)
    recommend(TEST_DATA_FILE, output_file, "emotion", "empath", 1.0, 1.0)

    recommend(TEST_DATA_FILE, output_file, "emotion_intensity", "tf_idf", 1.0, 1.0)
    recommend(TEST_DATA_FILE, output_file, "emotion_intensity", "empath", 1.0, 1.0)

# run_correlation_matrix_combo_reduce(TEST_DATA_FILE, "recommendations/12-25_age_10_plus_highly_rated_books/correlation_matrix_combo_reduce")
