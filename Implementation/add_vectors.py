from Recomender_Helper.vector_helper import get_vector_by_isbn, cosine_similarity, average_vectors, concat
import json

def add_vectors_to_stem_emotion_data_files():
    in_file_paths = ["stem_emotion_relation/any_age_users_who_do_not_like_stem_books.jsonl", "stem_emotion_relation/any_age_users_who_like_stem_books.jsonl", "stem_emotion_relation/any_age_users_who_mix_like_stem_books.jsonl"]
    out_file_paths = ["stem_emotion_relation/any_age_users_who_do_not_like_stem_books_with_vec.jsonl", "stem_emotion_relation/any_age_users_who_like_stem_books_with_vec.jsonl", "stem_emotion_relation/any_age_users_who_mix_like_stem_books_with_vec.jsonl"]


    for in_file, out_file in zip(in_file_paths, out_file_paths):
        with open(in_file, "r") as f_in, open(out_file, "w") as f_out:
            for line in f_in:
                user = json.loads(line)
                user_vecs = []
                book_list = user["book_list"]

                for book in book_list:
                    isbn = book["isbn"]
                    emotion_intensity = get_vector_by_isbn(isbn, "emotion_intensity")
                    book["emotion_intensity"] = emotion_intensity
                    if emotion_intensity is not None:
                        user_vecs.append(emotion_intensity)
                if user_vecs:
                    user["emotion_vec_all_books"] = average_vectors(user_vecs)
                else:
                    user["emotion_vec_all_books"] = None
                
                f_out.write(json.dumps(user) + "\n")
        