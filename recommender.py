import json
import os
from tqdm import tqdm
from collections import Counter
from Recomender_Helper.vector_helper import get_vector_by_isbn, cosine_similarity, average_vectors, concat, concat_with_weight
from Vectorizer.EmotionConditionedTopicVectorMaker import EmotionConditionedTopicVectorMaker

# ---- Paths ----

TEST_DATA_FILE = "user_eval_sets/users_1_plus_STEM_books_and_10_plus_high_rated_split_60_40.json"
#TEST_DATA_FILE = "user_eval_sets/test.json"
STEM_BOOKS_FILE = "processed_data/stem_isbns_from_topic.txt"

emotion_conditioned_vector_maker = EmotionConditionedTopicVectorMaker()


def handle_book(isbn, emotion_type, topic_type, emotion_weight = 1.0, topic_weight = 1.0):
    emotion_vec = get_vector_by_isbn(isbn, emotion_type)
    if emotion_vec is None:
        raise Exception(f"Do not have vector for {isbn}")
    topic_vec = get_vector_by_isbn(isbn, topic_type)

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

def make_candidate_profile(profile_books, emotion_type, general_stem_topic_vec, emotion_weight = 1.0, topic_weight = 1.0):
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



def recomend(test_data_file, emotion_type = "emotion_intensity", topic_type = "tf_idf", emotion_weight = 1.0, topic_weight = 1.0):
    general_stem_topic_vec = general_stem_topic_vec_maker(topic_type)
    with open(test_data_file, 'r') as file:
        data = json.load(file)
    
    for item in tqdm(data):
        profile_books = item['candidate_profile']
        try:
            candidate_profile = make_candidate_profile(profile_books, emotion_type, general_stem_topic_vec, emotion_weight, topic_weight)
            recomendation_books = item['recommendation_list']
            for book in recomendation_books:
                try:
                    book_vec = handle_book(book['isbn'], emotion_type, topic_type, emotion_weight, topic_weight)
                    #book_vec = get_vector_by_isbn(book['isbn'], emotion_type)
                    cos = cosine_similarity(candidate_profile, book_vec)
                    book['cos'] = cos
                except Exception as e:
                    print(e)
        except Exception as e:
            print(e)
            print(item['user_id'])
    
    output_file_name = f"recommendations/1_plus_stem_books_weight/{emotion_type}_with_weight_{emotion_weight}__{topic_type}_with_weight_{topic_weight}.json"
    with open(output_file_name, 'w') as f:
        json.dump(data, f, indent=4)


def test_weights(emotion_type="emotion", topic_type="tf_idf"):
    for i in range(10):
        e_weight = (i + 1) / 10
        t_weight = 1.0 - (i / 10)

        recomend(TEST_DATA_FILE, emotion_type, topic_type, emotion_weight=e_weight, topic_weight=t_weight)


recomend(TEST_DATA_FILE, emotion_type="emotion", topic_type="empath", emotion_weight = 0.001, topic_weight = 1.0)

#test_weights(topic_type="sentance_bert")