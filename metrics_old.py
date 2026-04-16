import json
import numpy as np
import math
import csv
import os
from sklearn.metrics import ndcg_score
import scipy.stats as stats
from scipy.stats import wilcoxon

def get_rr(y_true, threshold=7):
    x = 1
    for y in y_true:
        if y >= threshold:
            return 1.0/x
        x+=1
    return 0.0


def precision_at_k(ranked_ratings, k, threshold=7):
    top_k = ranked_ratings[:k]
    if len(top_k) == 0:
        return 0.0
    relevant = sum(1 for r in top_k if r >= threshold)
    return relevant / k

def boost_stem_ratings(user_obj, boost_amount):
    recommendations = user_obj.get("recommendation_list", [])

    for item in recommendations:
        if item.get("is_stem") is True:
            item["rating"] += boost_amount

    return user_obj

def handle_user(record):
    record = boost_stem_ratings(record, 2)
    recommendations = record["recommendation_list"]

    filtered = [rec for rec in recommendations if rec["rating"] != 0]
    # The filtering should be redundent but just in case 

    y_score = np.array([[rec["cos"] for rec in filtered]])
    y_true = np.array([[rec["rating"] for rec in filtered]])

    ndcg = ndcg_score(y_true, y_score)
    ndcg_5 = ndcg_score(y_true, y_score, k=5)
    ndcg_10 = ndcg_score(y_true, y_score, k=10)

    recommendations_sorted = sorted(filtered, key=lambda r: r["cos"], reverse=True)
    ranked_ratings = [rec["rating"] for rec in recommendations_sorted]
    rr = get_rr(ranked_ratings)

    scores = [rec["cos"] for rec in filtered]
    ratings = [rec["rating"] for rec in filtered]
    rho, p_value = stats.spearmanr(scores, ratings)
    if math.isnan(rho):
        print(record["user_id"])

    # Precision@K
    p1 = precision_at_k(ranked_ratings, k=1)
    p3 = precision_at_k(ranked_ratings, k=3)
    p5 = precision_at_k(ranked_ratings, k=5)

    return ndcg, ndcg_5, ndcg_10, rho, p_value, rr, p1, p3, p5


def evaluate_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    ndcg_list = []
    ndcg_5_list = []
    ndcg_10_list = []
    rho_list = []
    rr_list = []
    p1_list = []
    p3_list = []
    p5_list = []

    for record in records:
        ndcg, ndcg_5, ndcg_10, rho, p_value, rr, p1, p3, p5 = handle_user(record)

        ndcg_list.append(ndcg)
        ndcg_5_list.append(ndcg_5)
        ndcg_10_list.append(ndcg_10)
        rho_list.append(rho)
        rr_list.append(rr)
        p1_list.append(p1)
        p3_list.append(p3)
        p5_list.append(p5)

    return {
        "ndcg": ndcg_list,
        "ndcg5": ndcg_5_list,
        "ndcg10": ndcg_10_list,
        "rho": rho_list,
        "mrr": rr_list,
        "p1": p1_list,
        "p3": p3_list,
        "p5": p5_list
    }

def general_metrics(file_path, should_print=True):
    results = evaluate_file(file_path)
    ndcg_list = results["ndcg"]
    ndcg_5_list = results["ndcg5"]
    ndcg_10_list = results["ndcg10"]
    rho_list = results["rho"]
    rr_list = results["mrr"]
    p1_list = results["p1"]
    p3_list = results["p3"]
    p5_list = results["p5"]


    overall_ndcg = np.mean(ndcg_list)
    overall_ndcg_5 = np.mean(ndcg_5_list)
    overall_ndcg_10 = np.mean(ndcg_10_list)
    clean_rho = [x for x in rho_list if x is not None and not math.isnan(x)]
    overall_rho = np.mean(clean_rho)
    mrr = np.mean(rr_list)
    overall_p1 = np.mean(p1_list)
    overall_p3 = np.mean(p3_list)
    overall_p5 = np.mean(p5_list)

    if should_print:
        print(file_path)
        print("Overall NDCG:", overall_ndcg)
        print("Overall NDCG@5:", overall_ndcg_5)
        print("Overall NDCG@10:", overall_ndcg_10)
        print("Overall Spearman rho:", overall_rho)
        print("MRR:", mrr)
        print("P@1:", overall_p1)
        print("P@3:", overall_p3)
        print("P@5:", overall_p5)
    else:
        return overall_ndcg, overall_ndcg_5, overall_ndcg_10, overall_rho, mrr, overall_p1, overall_p3, overall_p5


def run_wilcoxon(file_A, file_B, should_print=True):
    results_A = evaluate_file(file_A)
    results_B = evaluate_file(file_B)
    metrics = ["ndcg", "ndcg5", "ndcg10", "rho", "mrr", "p1", "p3", "p5"]

    results = []

    for metric in metrics:
        if metric == "rho":
            rho_A = np.array(results_A["rho"])
            rho_B = np.array(results_B["rho"])
            mask = ~np.isnan(rho_A) & ~np.isnan(rho_B)
            A = rho_A[mask]
            B = rho_B[mask]
        else:
            A = results_A[metric]
            B = results_B[metric]
        stat, p = wilcoxon(A, B, alternative="greater")
        results.append((metric, stat, p))
        
    if should_print:
        for metric, stat, p in results:
            print(f"{metric}: statistic={stat:.4f}, p={p:.6f}")
    else:
        return results


def metrics_to_csv(file_paths_and_names, output_csv):
    rows = []

    for file_path, names in file_paths_and_names:
        ndcg, ndcg5, ndcg10, rho, mrr, p1, p3, p5 = general_metrics(
            file_path, should_print=False
        )

        rows.append([
            names,
            ndcg,
            ndcg5,
            ndcg10,
            rho,
            mrr,
            p1,
            p3,
            p5
        ])

    header = [
        "Type",
        "ndcg",
        "ndcg@5",
        "ndcg@10",
        "spearman_rho",
        "mrr",
        "p@1",
        "p@3",
        "p@5"
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

def compare_files_and_save(file_A, file_B_list, output_csv):
    metrics = ["ndcg", "ndcg5", "ndcg10", "rho", "mrr", "p1", "p3", "p5"]
    rows = []

    for file_B in file_B_list:
        results = run_wilcoxon(file_A[0], file_B[0], should_print=False)
        
        # Create a dictionary for this specific comparison
        # We start with the comparison label
        row = {
            "comparison": f"{file_A[1]} vs {file_B[1]}"
        }
        
        # Map each metric result to its corresponding column
        # results is a list of (metric, stat, p)
        for metric, stat, p in results:
            row[metric] = p
            
        rows.append(row)

    with open(output_csv, "w", newline="") as f:
        # Define fieldnames: the comparison label + all metrics
        fieldnames = ["comparison"] + metrics
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(rows)

def set_up_to_save_wilcoxon():
    A = ("recommendations/1_2_3_stem_books/recommendation_using_emotion_tf_idf.json", "Emotion and TF-IDF")

    type_list = [
        ("recommendation_using_emotion_empath.json", "Emotion and Empath"),
        ("recommendation_using_emotion_tf_idf.json", "Emotion and TF-IDF"),
        ("recommendation_using_emotion_intensity_doc2vec.json", "Emotion Intensity and Doc2Vec"),
        ("recommendation_using_emotion_doc2vec.json", "Emotion and Doc2Vec"),
        ("recommendation_using_emotion_intensity_empath.json", "Emotion Intensity and Empath"),
        ("recommendation_using_emotion_intensity_tf_idf.json", "Emotion Intensity and TF-IDF"),
        ("recommendation_using_emotion_intensity_word2vec.json", "Emotion Intensity and Word2Vec"),
        ("recommendation_using_emotion_word2vec.json", "Emotion and Word2Vec"),
        ("recommendation_using_emotion_intensity_glove.json", "Emotion Intensity and GloVE"),
        ("recommendation_using_emotion_intensity.json", "Emotion Intensity no topic"),
        ("recommendation_using_emotion.json", "Emotion no topic"),
    ]

    file_paths_and_names = []
    for file,title in type_list:
        temp = f"recommendations/1_2_3_stem_books/{file}"
        file_paths_and_names.append((temp, title))
    
    compare_files_and_save(A, file_paths_and_names, "recommendations/1_2_3_stem_books/wilcoxon_comparison.csv")

def set_up_to_save():
    type_list = [
        ("recommendation_using_emotion_empath.json", "Emotion and Empath"),
        ("recommendation_using_emotion_tf_idf.json", "Emotion and TF-IDF"),
        ("recommendation_using_emotion_intensity_doc2vec.json", "Emotion Intensity and Doc2Vec"),
        ("recommendation_using_emotion_doc2vec.json", "Emotion and Doc2Vec"),
        ("recommendation_using_emotion_intensity_empath.json", "Emotion Intensity and Empath"),
        ("recommendation_using_emotion_intensity_tf_idf.json", "Emotion Intensity and TF-IDF"),
        ("recommendation_using_emotion_intensity_word2vec.json", "Emotion Intensity and Word2Vec"),
        ("recommendation_using_emotion_word2vec.json", "Emotion and Word2Vec"),
        ("recommendation_using_emotion_intensity_glove.json", "Emotion Intensity and GloVE"),
        ("recommendation_using_emotion_intensity.json", "Emotion Intensity no topic"),
        ("recommendation_using_emotion.json", "Emotion no topic"),
    ]

    file_paths_and_names = []
    for file,title in type_list:
        temp = f"recommendations/1_2_3_stem_books/{file}"
        file_paths_and_names.append((temp, title))
    
    metrics_to_csv(file_paths_and_names, "recommendations/1_2_3_stem_books/comparison.csv")



# file_name_B = "recommendation_using_emotion_tf_idf.json"
# file_path_B = f"recommendations/{file_name_B}"

# run_wilcoxon(file_path_B, file_path_A)

#set_up_to_save_wilcoxon()