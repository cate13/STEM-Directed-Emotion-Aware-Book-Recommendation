import json
import os
import csv
import math
import random
import numpy as np
from sklearn.metrics import ndcg_score
import scipy.stats as stats
from scipy.stats import wilcoxon
import re
from tqdm import tqdm 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CURATED_USERS_PATH = os.path.join(
    BASE_DIR, "processed_data", "curated_users.jsonl"
)

def _load_stem_isbns(stem_paths):
    """Load all STEM ISBNs into a set for fast lookup."""
    stem_isbns = set()

    for path in stem_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                stem_isbns.add(line.strip())

    return stem_isbns

STEM_ISBNS = _load_stem_isbns(["processed_data/stem_isbns_from_classifier.txt", "processed_data/stem_isbns_from_cosine.txt", "processed_data/stem_isbns_from_classifier.txt"])

def relevant(isbn):
    if isbn in STEM_ISBNS:
        return True 
    return False

def get_rr(book_list):
    x = 1
    for book in book_list:
        if relevant(book):
            return 1.0/x
        x+=1
    return 0.0

def precision_at_k(ranked_ratings, k):
    if k <= 0:
        return 0.0
    top_k_items = ranked_ratings[:k]
    relevant_count = sum(1 for item in top_k_items if relevant(item))
    return relevant_count / k

def handle_user(user):
    recommendations = user["recommendation_list"]
    filtered = [rec for rec in recommendations if rec["rating"] != 0]
    # The filtering should be redundent but just in case 

    recommendations_sorted = sorted(filtered, key=lambda r: r["cos"], reverse=True)
    book_id_for_recommendations_sorted = [rec["isbn"] for rec in recommendations_sorted]
   
    rr = get_rr(book_id_for_recommendations_sorted)
    p1 = precision_at_k(book_id_for_recommendations_sorted, k=1)
    p3 = precision_at_k(book_id_for_recommendations_sorted, k=3)
    p5 = precision_at_k(book_id_for_recommendations_sorted, k=5)

    # scores = [rec["cos"] for rec in filtered]
    # ratings = [rec["rating"] for rec in filtered]
    # rho, p_value = stats.spearmanr(scores, ratings)

    # y_score = np.array([scores]) 
    # y_true = np.array([ratings])

    # ndcg_2 = ndcg_score(y_true, y_score, k=2)
    # ndcg_3 = ndcg_score(y_true, y_score, k=3)
    # ndcg_5 = ndcg_score(y_true, y_score, k=5)

    # return rr, p1, p3, p5, rho, ndcg_2, ndcg_3, ndcg_5

    only_stem = [item for item in filtered if item.get('is_stem') is True]
    if len(only_stem) >= 2:
        scores = [rec["cos"] for rec in only_stem]
        ratings = [rec["rating"] for rec in only_stem]
        y_score = np.array([scores]) 
        y_true = np.array([ratings])

        ndcg_2 = ndcg_score(y_true, y_score, k=2)
        ndcg_3 = None
        ndcg_5 = None
        ndcg_10 = None
        rho = None


        if len(only_stem) >= 3:
            ndcg_3 = ndcg_score(y_true, y_score, k=3)

            if len(only_stem) >= 5:
                ndcg_5 = ndcg_score(y_true, y_score, k=5)

                if len(only_stem) >= 10:
                    ndcg_10 = ndcg_score(y_true, y_score, k=10)
                    rho, _ = stats.spearmanr(scores, ratings)
        return rr, p1, p3, p5, (abs(rho) if rho is not None else None), ndcg_2, ndcg_3, ndcg_5, ndcg_10
    else:
        return rr, p1, p3, p5, None, None, None, None, None


def evaluate_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    rr_list = []
    p1_list = []
    p3_list = []
    p5_list = []
    spearman_list = []
    ndcg_2_list = []
    ndcg_3_list = []
    ndcg_5_list = []
    ndcg_10_list = []

    for record in records:
        rr, p1, p3, p5, rho, ndcg_2, ndcg_3, ndcg_5, ndcg_10 = handle_user(record)

        rr_list.append(rr)
        p1_list.append(p1)
        p3_list.append(p3)
        p5_list.append(p5)

        spearman_list.append(rho)
        ndcg_2_list.append(ndcg_2)
        ndcg_3_list.append(ndcg_3)
        ndcg_5_list.append(ndcg_5)
        ndcg_10_list.append(ndcg_10)

    return {
        "mrr" : rr_list,
        "P@1" : p1_list,
        "P@3" : p3_list,
        "P@5" : p5_list,        
        "Spearman" : spearman_list,
        "NDCG_2" : ndcg_2_list,
        "NDCG_3" : ndcg_3_list,
        "NDCG_5" : ndcg_5_list,
        "NDCG_10" : ndcg_10_list,
    }

def calculate_metrics(file_path, is_llm = False):
    if is_llm:
        results = evaluate_llm_file(file_path)
    else:
        results = evaluate_file(file_path)
    rr_list = results["mrr"]
    p1_list = results["P@1"]
    p3_list = results["P@3"]
    p5_list = results["P@5"]

    spearman_list = results["Spearman"]
    ndcg_2_list = results["NDCG_2"]
    ndcg_3_list = results["NDCG_3"]
    ndcg_5_list = results["NDCG_5"]
    ndcg_10_list = results["NDCG_10"]

    mrr = np.mean(rr_list)
    overall_p1 = np.mean(p1_list)
    overall_p3 = np.mean(p3_list)
    overall_p5 = np.mean(p5_list)

    clean_spearman = [x for x in spearman_list if x is not None and not math.isnan(x)]
    spearman = np.mean(clean_spearman)

    clean_ndcg_2 = [x for x in ndcg_2_list if x is not None and not math.isnan(x)]
    ndcg_2 = np.mean(clean_ndcg_2)

    clean_ndcg_3 = [x for x in ndcg_3_list if x is not None and not math.isnan(x)]
    ndcg_3 = np.mean(clean_ndcg_3)

    clean_ndcg_5 = [x for x in ndcg_5_list if x is not None and not math.isnan(x)]
    ndcg_5 = np.mean(clean_ndcg_5)

    clean_ndcg_10 = [x for x in ndcg_10_list if x is not None and not math.isnan(x)]
    ndcg_10 = np.mean(clean_ndcg_10)

    return {
        "mrr" : mrr,
        "P@1" : overall_p1,
        "P@3" : overall_p3,
        "P@5" : overall_p5,
        "Spearman" : spearman,
        "NDCG_2" : ndcg_2,
        "NDCG_3" : ndcg_3,
        "NDCG_5" : ndcg_5,
        "NDCG_10" : ndcg_10,
    }
    
def get_file_names(target_folder):
    files = [f for f in os.listdir(target_folder) if os.path.isfile(os.path.join(target_folder, f))]
    all_files = [f for f in os.listdir(target_folder) if f.endswith('.json')]
    for filename in all_files:
        print(filename)

def file_name_to_column_entry(file_name):
    name = file_name.replace('.json', '')
    name = name.replace('tf_idf', 'TF-IDF')
    name = name.replace('_', ' ')
    return name.capitalize()

def aggregate_metrics_to_csv(target_folder, output_filename="results.csv"):
    # 1. Gather all files (filtering for .txt or .json as needed)
    files = [f for f in os.listdir(target_folder) if os.path.isfile(os.path.join(target_folder, f))]
    
    if not files:
        print("No files found in the specified directory.")
        return

    all_files = [f for f in os.listdir(target_folder) if f.endswith('.json')]

    all_rows = []
    
    # 2. Process each file
    for filename in tqdm(all_files):
        file_path = os.path.join(target_folder, filename)
        
        try:
            # Get the metrics dict from your existing function
            metrics = calculate_metrics(file_path)
            
            # Insert the filename at the start of the dictionary
            # This ensures the row label is included
            row = {"Model": file_name_to_column_entry(file_name=filename)}
            row.update(metrics)
            all_rows.append(row)
        except Exception as e:
            print(f"Skipping {filename} due to error: {e}")

    # 3. Define headers based on the first result
    if all_rows:
        headers = all_rows[0].keys()

        out_file_path = os.path.join(target_folder, output_filename)
        
        with open(out_file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(all_rows)
            
        print(f"Successfully wrote {len(all_rows)} rows to {output_filename}")


def compare_baseline_to_others(target_folder, baseline_filename):
    baseline_path = os.path.join(target_folder, baseline_filename)
    if not os.path.exists(baseline_path):
        print(f"Baseline file {baseline_filename} not found.")
        return

    # Get baseline lists
    baseline_data = evaluate_file(baseline_path)
    
    # 1. FILTER: Only look at .json files and ignore our own output CSVs
    all_files = [f for f in os.listdir(target_folder) 
                 if f.endswith('.json') and f != baseline_filename]

    results_rows = []

    for comp_file in tqdm(all_files):
        comp_path = os.path.join(target_folder, comp_file)
        try:
            comp_data = evaluate_file(comp_path)
            
            row = {"Comparison File": file_name_to_column_entry(comp_file)}

            for metric in baseline_data.keys():
                base_scores = baseline_data[metric]
                comp_scores = comp_data[metric]

                # 2. ROBUST CLEANING: Remove None AND NaN values
                # This ensures Spearman NaNs don't break the Wilcoxon test
                paired_scores = []
                for b, c in zip(base_scores, comp_scores):
                    if b is not None and c is not None:
                        # Check if either is NaN (using math.isnan)
                        if not (math.isnan(b) or math.isnan(c)):
                            paired_scores.append((b, c))
                
                if len(paired_scores) < 10: # Wilcoxon needs a decent sample size
                    row[f"{metric}_p_val"] = "Insufficient Data"
                    continue

                b_clean, c_clean = zip(*paired_scores)

                # Check if every single pair is identical
                if np.array_equal(b_clean, c_clean):
                    row[f"{metric}_p_val"] = 1.0
                else:
                    # 'greater' tests if baseline > comparison
                    # zero_method="pratt" or "wilcox" handles ties
                    _, p_val = wilcoxon(b_clean, c_clean, alternative='greater')
                    # 2. Use scientific notation or more decimals if you don't want 0.0
                    row[f"{metric}_p_val"] = "{:.2e}".format(p_val) if p_val < 0.0001 else round(p_val, 5)

            results_rows.append(row)

        except Exception as e:
            print(f"Skipping {comp_file} due to error: {e}")

    # Write results
    if results_rows:
        output_file_name = f"Wilcoxon Compared to {file_name_to_column_entry(baseline_filename)}.csv"
        out_path = os.path.join(target_folder, output_file_name)
        headers = results_rows[0].keys()
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results_rows)
        print(f"Successfully wrote Wilcoxon results to {output_file_name}")


def normalize(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text

def parse_llm_ranking(llm_text):
    lines = llm_text.split("\n")
    results = []

    for line in lines:
        match = re.match(r"\d+\.\s+(.*)", line)
        if match:
            full = match.group(1).strip()

            # Split "Title by Author"
            if " by " in full:
                title, author = full.rsplit(" by ", 1)
            else:
                title, author = full, None

            results.append({
                "title": title.strip(),
                "author": author.strip() if author else None
            })

    return results

def build_book_lookup(user_data):
    title_author_lookup = {}
    title_lookup = {}

    for book in user_data["recommendation_list"]:
        norm_title = normalize(book["title"])
        norm_author = normalize(book["author"])

        title_author_lookup[(norm_title, norm_author)] = book

        # allow multiple books per title (rare but safer)
        if norm_title not in title_lookup:
            title_lookup[norm_title] = []
        title_lookup[norm_title].append(book)

    return title_author_lookup, title_lookup

def match_book(item, title_author_lookup, title_lookup):
    norm_title = normalize(item["title"])
    norm_author = normalize(item["author"])

    # 1. exact match (title + author)
    if norm_author and (norm_title, norm_author) in title_author_lookup:
        return title_author_lookup[(norm_title, norm_author)]

    # 2. title-only match
    if norm_title in title_lookup:
        return title_lookup[norm_title][0]  # take first match

    return None

def convert_llm_to_recommendations(llm_json, user_data, user_id):
    parsed = parse_llm_ranking(llm_json["llm_ranking"])
    title_author_lookup, title_lookup = build_book_lookup(user_data)

    k = len(user_data["recommendation_list"])
    parsed = parsed[:k]

    recommendations = []

    for i, item in enumerate(parsed):
        book = match_book(item, title_author_lookup, title_lookup)

        if book:
            recommendations.append({
                "isbn": book["isbn"],
                "rating": book.get("rating", 0),
                "cos": k - i,
                "is_stem": book.get("is_stem", False)
            })
        else:
            recommendations.append({
                "isbn": None,
                "rating": 0,
                "cos": k - i,
                "is_stem": False
            })

    return {
        "user_id": llm_json["user_id"],
        "recommendation_list": recommendations
    }

def evaluate_llm_file(llm_path):
    rr_list = []
    p1_list = []
    p3_list = []
    p5_list = []
    spearman_list = []
    ndcg_2_list = []
    ndcg_3_list = []
    ndcg_5_list = []
    ndcg_10_list = []
    with open(llm_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    
    with open("enriched_user_profiles.json", "r", encoding="utf-8") as f_2:
        user_start = json.load(f_2)

    len1, len2 = len(user_start), len(records)
    if len1 != len2:
        print(f"Warning: File lengths do not match! (File 1: {len1}, File 2: {len2})")
        return 
    
    for i in range(len1):
        user_data_id = user_start[i].get('user_id')
        llm_output_user_id = records[i].get('user_id')

        if user_data_id != llm_output_user_id:
            print("user ids don't match {user_data_id} {llm_output_user_id}")
        else:
            rr, p1, p3, p5, rho, ndcg_2, ndcg_3, ndcg_5, ndcg_10 = handle_user(convert_llm_to_recommendations(records[i], user_start[i], user_data_id))
            rr_list.append(rr)
            p1_list.append(p1)
            p3_list.append(p3)
            p5_list.append(p5)

            spearman_list.append(rho)
            ndcg_2_list.append(ndcg_2)
            ndcg_3_list.append(ndcg_3)
            ndcg_5_list.append(ndcg_5)
            ndcg_10_list.append(ndcg_10)

    return {
        "mrr" : rr_list,
        "P@1" : p1_list,
        "P@3" : p3_list,
        "P@5" : p5_list,        
        "Spearman" : spearman_list,
        "NDCG_2" : ndcg_2_list,
        "NDCG_3" : ndcg_3_list,
        "NDCG_5" : ndcg_5_list,
        "NDCG_10" : ndcg_10_list,
    }


def compare_baseline_to_others_llm():
    llm = "mistral"
    baseline_path = f"recommendations/1_plus_stem_books/with_stem_vec/user_rankings_result_{llm}.json"

    # Get baseline lists 
    baseline_data = evaluate_llm_file(baseline_path)

    # 1. FILTER: Only look at .json files and ignore our own output CSVs
    all_files = [
        f"recommendations/1_plus_stem_books/straight_no_stem/user_rankings_results_{llm}.json",
        f"recommendations/1_plus_stem_books/with_stem_instructions/user_rankings_result_{llm}.json",
        f"recommendations/1_plus_stem_books/with_stem_and_emotion/user_rankings_result_{llm}.json"
    ]

    results_rows = []

    for comp_path in all_files:
        try:
            comp_data = evaluate_llm_file(comp_path)
            row = {"Comparison File": comp_path}
            
            for metric in baseline_data.keys():
                base_scores = baseline_data[metric]
                comp_scores = comp_data[metric]

                # 2. ROBUST CLEANING: Remove None AND NaN values
                # This ensures Spearman NaNs don't break the Wilcoxon test
                paired_scores = []
                for b, c in zip(base_scores, comp_scores):
                    if b is not None and c is not None:
                        # Check if either is NaN (using math.isnan)
                        if not (math.isnan(b) or math.isnan(c)):
                            paired_scores.append((b, c))
                
                if len(paired_scores) < 10: # Wilcoxon needs a decent sample size
                    row[f"{metric}_p_val"] = "Insufficient Data"
                    continue

                b_clean, c_clean = zip(*paired_scores)

                # Check if every single pair is identical
                if np.array_equal(b_clean, c_clean):
                    row[f"{metric}_p_val"] = 1.0
                else:
                    # 'greater' tests if baseline > comparison
                    # zero_method="pratt" or "wilcox" handles ties
                    _, p_val = wilcoxon(b_clean, c_clean, alternative='greater')
                    # 2. Use scientific notation or more decimals if you don't want 0.0
                    row[f"{metric}_p_val"] = "{:.2e}".format(p_val) if p_val < 0.0001 else round(p_val, 5)

            results_rows.append(row)

        except Exception as e:
            print(f"Skipping {comp_path} due to error: {e}")

    # Write results
    if results_rows:
        
        out_path = f"recommendations/1_plus_stem_books/{llm}_Wilcoxon.csv"
        headers = results_rows[0].keys()
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results_rows)
        print(f"Successfully wrote Wilcoxon results to {out_path}")



compare_baseline_to_others(target_folder="recommendations/12-25_age_10_plus_highly_rated_books/sample_comparison", baseline_filename="emotion_intensity_with_weight_0.1_empath_with_weight_0.9.json")


# aggregate_metrics_to_csv(target_folder="recommendations/12-25_age_10_plus_highly_rated_books/sample_comparison")

