# Create data
import scipy.stats as stats
import pandas as pd

## Version 1 - Array
# split = [0.09523809523809525, 0, 0, 0.09523809523809525, 0, 0.09523809523809525, 0, 0, 0.09523809523809525, 0]
# other = [0, 0, 0.09523809523809525, 0.09523809523809525, 0, 0, 0, 0, 0, 0]

# conduct the Wilcoxon-Signed Rank Test
# print(stats.wilcoxon(split, other))
# if less than 0.05, is significantly different

## Version 2 - CSV
def list_pad(list1, list2):
    nlist1 = list1.values.tolist()
    nlist2 = list2.values.tolist()
    return array_pad(nlist1, nlist2)

## Version 2 - CSV
def array_pad(list1, list2):
    if len(list1) > len(list2):
        while len(list1) > len(list2):
            list2.append(0.0)
    elif len(list1) < len(list2):
        while len(list1) < len(list2):
            list1.append(0.0)
    return list1, list2

def compute_rouge_wilcoxon(data1, data2):
    array1, array2 = list_pad(data1['rouge1_precision'], data2['rouge1_precision'])
    print('rouge1_precision:', stats.wilcoxon(array1, array2).pvalue)
    array1, array2 = list_pad(data1['rouge1_recall'], data2['rouge1_recall'])
    print('rouge1_recall:', stats.wilcoxon(array1, array2).pvalue)
    array1, array2 = list_pad(data1['rouge1_f1'], data2['rouge1_f1'])
    print('rouge1_f1:', stats.wilcoxon(array1, array2).pvalue)
    array1, array2 = list_pad(data1['rouge2_precision'], data2['rouge2_precision'])
    print('rouge2_precision:', stats.wilcoxon(array1, array2).pvalue)
    array1, array2 = list_pad(data1['rouge2_recall'], data2['rouge2_recall'])
    print('rouge2_recall:', stats.wilcoxon(array1, array2).pvalue)
    array1, array2 = list_pad(data1['rouge2_f1'], data2['rouge2_f1'])
    print('rouge2_f1:', stats.wilcoxon(array1, array2).pvalue)
    array1, array2 = list_pad(data1['rougeL_precision'], data2['rougeL_precision'])
    print('rougeL_precision:', stats.wilcoxon(array1, array2).pvalue)
    array1, array2 = list_pad(data1['rougeL_recall'], data2['rougeL_recall'])
    print('rougeL_recall:', stats.wilcoxon(array1, array2).pvalue)
    array1, array2 = list_pad(data1['rougeL_f1'], data2['rougeL_f1'])
    print('rougeL_f1:', stats.wilcoxon(array1, array2).pvalue)

def compute_gruen_wilcoxon(data1, data2):
    array1, array2 = list_pad(data1['gruen_grammaticality_score'], data2['gruen_grammaticality_score'])
    print('gruen_grammaticality_score:', stats.wilcoxon(array1, array2).pvalue)
    array1, array2 = list_pad(data1['gruen_redundancy_score'], data2['gruen_redundancy_score'])
    print('gruen_redundancy_score:', stats.wilcoxon(array1, array2).pvalue)
    array1, array2 = list_pad(data1['gruen_focus_score'], data2['gruen_focus_score'])
    print('gruen_focus_score:', stats.wilcoxon(array1, array2).pvalue)
    array1, array2 = list_pad(data1['gruen_linear_combo_score'], data2['gruen_linear_combo_score'])
    print('gruen_linear_combo_score:', stats.wilcoxon(array1, array2).pvalue)

def wilcoxon_csv():
    subreddit = "askscience"
    # ROUGE
    # hybrid_path = f"../summaries/rouge_scores/{subreddit}_scores/{subreddit}_rouge_scores_bart_vs_original.csv"
    # method_path = f"../summaries/rouge_scores/{subreddit}_scores/{subreddit}_rouge_scores_bart_orig_vs_original.csv"
    # method_path = f"../summaries/rouge_scores/{subreddit}_scores/{subreddit}_rouge_scores_basis_vs_original.csv"
    # method_path = f"../summaries/rouge_scores/{subreddit}_scores/{subreddit}_rouge_scores_tfidf_vs_original.csv"
    # method_path = f"../summaries/rouge_scores/{subreddit}_scores/{subreddit}_rouge_scores_diversity_1_vs_original.csv"
    # method_path = f"../summaries/rouge_scores/{subreddit}_scores/{subreddit}_rouge_scores_diversity_2_vs_original.csv"
    # GRUEN
    hybrid_path = f"../summaries/gruen_scores/{subreddit}_scores/{subreddit}_gruen_scores_bart.csv"
    # method_path = f"../summaries/gruen_scores/{subreddit}_scores/{subreddit}_gruen_scores_bart_orig.csv"
    # method_path = f"../summaries/gruen_scores/{subreddit}_scores/{subreddit}_gruen_scores_basis.csv"
    # method_path = f"../summaries/gruen_scores/{subreddit}_scores/{subreddit}_gruen_scores_tfidf.csv"
    # method_path = f"../summaries/gruen_scores/{subreddit}_scores/{subreddit}_gruen_scores_diversity_1.csv"
    # method_path = f"../summaries/gruen_scores/{subreddit}_scores/{subreddit}_gruen_scores_diversity_2.csv"
    method_path = f"../summaries/gruen_scores/{subreddit}_scores/{subreddit}_gruen_scores_original.csv"


    data1 = pd.read_csv(hybrid_path)
    data2 = pd.read_csv(method_path)

    # compute_rouge_wilcoxon(data1, data2)
    compute_gruen_wilcoxon(data1, data2)

# Example:
def wilcoxon_array():
    # group1 = [0.2, 0.8, 1.0, 0.6, 0.6, 1.0, 0.2, 0.8, 0.8, 0.6, 0.8, 0.8, 1.0, 0.6, 0.4, 0.4, 1.0, 0.2, 0.6, 0.4, 1.0, 0.0, 0.0, 0.4, 0.6, 0.4, 1.0, 0.8, 0.2, 0.6, 0.6, 0.8, 0.4, 1.0, 0.0, 0.6, 0.0, 1.0, 0.8, 0.6, 0.8, 0.8, 0.4, 1.0, 0.0, 0.8, 1.0, 0.8, 1.0, 0.6, 0.6, 0.6, 0.8, 0.0, 0.6, 0.4, 0.8, 0.8, 0.6, 0.6, 1.0, 0.0, 0.0, 0.0, 0.6, 0.8, 0.4, 1.0, 0.4, 0.6, 1.0, 0.6, 0.4, 0.6, 0.6, 0.6, 0.2, 0.8, 1.0, 0.4, 0.4, 0.8, 0.6, 0.4, 1.0, 1.0, 0.6, 0.2, 0.8, 0.4, 1.0, 0.6, 0.8, 0.4, 1.0, 0.6, 0.0, 0.8, 0.6, 1.0, 1.0, 0.8, 0.2, 0.4, 0.2, 0.0, 0.2, 0.4, 0.8, 0.6, 0.6, 0.6, 0.4, 0.8, 1.0, 0.4, 0.4, 0.4, 0.8, 0.2, 1.0, 0.4, 0.4, 0.2, 0.4, 1.0, 0.6, 0.0, 0.8, 0.6, 0.8, 1.0, 0.6, 0.6, 0.8, 0.6, 0.4, 0.6, 0.4, 1.0, 0.6, 0.8, 0.6, 0.6, 0.4, 0.8, 0.4, 0.0, 0.6, 1.0, 0.6, 0.4, 0.0, 0.8, 0.6, 0.2, 0.4, 0.0]
    # group1 = [0.4, 0.4, 0.6, 0.6, 0.8, 0.4, 1.0, 0.8, 0.8, 0.8, 0.4, 0.4, 0.6, 0.2, 0.4, 0.2, 0.0, 0.4, 0.8, 0.6, 0.4, 0.6, 0.6, 0.8, 0.2, 0.4, 0.8, 1.0, 0.4, 0.8, 0.0, 1.0, 0.6, 0.6, 1.0, 0.0, 0.0, 0.8, 0.8, 0.4, 1.0, 0.0, 0.2, 1.0, 0.8, 0.2, 0.4, 0.0, 0.2, 0.4, 0.6, 0.8, 0.6, 0.8, 0.4, 1.0, 0.6, 0.8, 0.0, 0.2, 0.8, 0.4, 0.4, 0.4, 0.6, 1.0, 0.8, 0.6, 0.6, 0.4, 0.2, 0.4, 0.2, 0.6, 0.6, 0.8, 0.8, 0.8, 1.0, 0.2, 0.8, 0.4, 0.8, 0.8, 0.6, 0.2, 0.4, 0.0, 0.6, 1.0, 0.2, 0.0, 1.0, 0.8, 0.6, 0.4, 1.0, 0.4, 0.8, 0.4, 0.0, 0.6, 0.6, 0.6, 0.0, 0.4, 1.0, 0.2, 0.6, 1.0, 0.6, 0.4, 1.0, 0.6, 1.0, 0.6, 0.4, 0.6, 0.4, 0.2, 0.0, 0.2, 0.0, 0.6, 0.0, 0.4, 0.0, 1.0, 0.2, 0.6, 0.8, 0.2, 0.6, 0.4, 0.2, 0.6, 0.2, 0.8, 0.8, 0.4, 0.6, 0.8, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.4, 0.0, 0.8, 0.4]
    group1 = [0.4, 0.6, 1.0, 0.8, 1.0, 1.0, 1.0, 0.8, 0.4, 1.0, 1.0, 0.6, 1.0, 1.0, 0.6, 0.4, 1.0, 0.4, 0.8, 1.0, 1.0, 0.8, 1.0, 1.0, 1.0, 0.8, 0.6, 1.0, 0.6, 1.0, 0.4, 1.0, 0.6, 1.0, 1.0, 0.8, 0.2, 0.8, 1.0, 1.0, 1.0, 0.8, 0.4, 1.0, 0.8, 0.8, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.6, 1.0, 0.8, 1.0, 0.6, 0.8, 0.4, 0.8, 1.0, 0.4, 0.2, 0.6, 0.8, 1.0, 0.8, 0.8, 0.8, 0.8, 1.0, 0.4, 1.0, 0.4, 0.6, 1.0, 0.8, 0.6, 0.8, 1.0, 0.8, 0.8, 0.8, 0.8, 1.0, 0.8, 1.0, 0.6, 1.0, 1.0, 1.0, 0.4, 1.0, 0.8, 1.0, 0.6, 0.8, 1.0, 0.8, 1.0, 0.8, 1.0, 0.4, 0.4, 0.4, 0.6, 0.6, 1.0, 1.0, 1.0, 0.8, 0.6, 1.0, 0.8, 0.6, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.8, 0.8, 0.8, 1.0, 0.4, 0.4, 0.8, 1.0, 0.4, 1.0, 0.6, 0.6, 0.8, 0.8, 0.6, 0.8, 1.0, 0.6, 0.4, 1.0, 1.0, 1.0, 0.8, 0.2, 0.8, 1.0, 0.8, 0.8, 0.8, 0.4, 0.8, 1.0, 1.0, 0.2, 1.0, 0.6]
    
    # group 1 = greater than group 2
    # group1 = [0.4, 1.0, 1.0, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.6, 1.0, 1.0, 1.0, 0.6, 1.0, 1.0, 0.8, 1.0, 1.0, 0.8, 0.6, 1.0, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.6, 0.6, 1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.2, 1.0, 0.6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.6, 1.0, 0.8, 1.0, 0.6, 1.0, 1.0, 0.8, 1.0, 1.0, 1.0, 0.6, 0.8, 1.0, 1.0, 0.6, 0.8, 0.8, 1.0, 0.8, 1.0, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 1.0, 0.8, 0.4, 0.8, 1.0, 0.8, 1.0, 1.0, 1.0, 0.8, 0.8, 1.0, 1.0, 0.4, 0.6, 1.0, 0.6, 1.0, 0.8, 1.0, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 1.0, 0.8, 0.2, 0.4, 1.0, 0.8, 0.8, 1.0, 0.6, 1.0, 0.6, 1.0, 1.0, 0.6, 0.8, 0.8, 0.8, 1.0, 1.0, 0.0, 0.8, 1.0, 1.0, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 1.0, 1.0, 1.0, 0.8, 0.6, 0.8, 0.6, 0.8, 1.0, 1.0, 0.6, 1.0, 1.0, 1.0, 0.8, 1.0, 0.8, 0.8, 1.0, 1.0, 0.4, 1.0, 1.0, 1.0, 0.4, 0.8, 0.8, 0.6, 0.2, 0.6, 0.8, 1.0, 1.0, 1.0]
    
    # group2 = [0.2, 0.0, 0.0, 0.8, 0.8, 1.0, 0.0, 0.2, 0.8, 0.4, 0.0, 0.4, 0.2, 0.2, 0.4, 0.0, 0.4, 0.0, 0.6, 0.4, 0.6, 0.0, 0.0, 0.6, 0.0, 0.2, 0.6, 1.0, 0.6, 0.4, 0.0, 1.0, 0.4, 0.6, 0.0, 0.6, 0.4, 0.2, 0.6, 0.6, 0.6, 0.4, 0.2, 0.4, 0.2, 0.2, 1.0, 0.4, 0.4, 0.6, 1.0, 0.0, 0.8, 0.0, 0.2, 0.2, 1.0, 0.2, 0.4, 0.6, 0.6, 0.2, 0.0, 0.6, 0.4, 0.8, 0.0, 0.2, 0.4, 0.0, 0.8, 0.0, 0.2, 0.8, 0.4, 0.8, 0.8, 0.8, 1.0, 0.6, 0.2, 0.4, 0.0, 0.0, 0.6, 0.0, 0.4, 0.6, 1.0, 0.2, 0.4, 0.4, 0.8, 0.4, 0.8, 0.6, 0.8, 0.4, 0.0, 0.8, 0.6, 0.8, 0.0, 0.6, 0.0, 0.0, 0.8, 0.8, 0.4, 0.8, 0.2, 0.6, 0.6, 1.0, 1.0, 0.0, 0.4, 0.6, 0.4, 0.4, 0.8, 0.2, 0.4, 0.2, 0.4, 0.0, 1.0, 0.8, 0.4, 0.2, 0.6, 0.6, 0.6, 0.6, 0.4, 0.4, 0.6, 0.6, 0.6, 0.8, 0.4, 1.0, 0.8, 0.8, 0.8, 0.4, 0.4, 0.6, 0.4, 0.4, 0.4, 0.4, 0.2, 0.6, 0.4, 0.2, 0.6, 0.4]
    # group2 = [0.6, 0.2, 1.0, 0.6, 0.2, 0.8, 1.0, 0.0, 0.2, 0.4, 0.6, 0.2, 0.0, 1.0, 0.0, 0.2, 0.0, 0.4, 0.6, 0.8, 0.6, 0.6, 1.0, 0.6, 1.0, 0.4, 0.6, 0.6, 0.0, 0.8, 0.4, 0.0, 0.6, 0.8, 1.0, 0.2, 0.2, 0.6, 0.6, 0.2, 1.0, 0.0, 0.6, 0.2, 0.6, 1.0, 0.4, 0.2, 0.2, 0.6, 0.8, 0.8, 0.4, 0.8, 0.4, 0.4, 0.4, 0.6, 0.0, 0.6, 0.4, 0.4, 0.6, 0.2, 0.8, 1.0, 0.2, 0.2, 0.2, 1.0, 1.0, 0.6, 0.6, 0.6, 0.2, 0.8, 0.8, 0.6, 0.6, 0.0, 0.0, 0.2, 0.6, 0.8, 1.0, 1.0, 0.6, 0.0, 0.8, 1.0, 0.8, 0.2, 0.4, 1.0, 0.8, 0.4, 1.0, 0.6, 0.4, 0.4, 0.4, 0.4, 0.2, 1.0, 0.0, 0.4, 0.8, 0.0, 0.6, 0.8, 0.4, 0.6, 0.2, 0.8, 1.0, 0.4, 0.6, 0.4, 0.8, 0.6, 0.6, 0.2, 0.8, 0.4, 0.4, 0.0, 0.2, 1.0, 0.0, 1.0, 0.2, 0.2, 0.4, 0.6, 0.8, 0.4, 0.4, 0.4, 1.0, 0.4, 0.6, 0.4, 0.6, 0.2, 0.6, 0.8, 0.6, 0.2, 0.2, 0.2, 0.2, 0.6, 0.2, 0.4, 0.6, 0.0, 1.0, 0.2]
    group2 = [0.4, 0.6, 1.0, 0.6, 0.6, 1.0, 1.0, 1.0, 0.6, 1.0, 1.0, 0.4, 1.0, 1.0, 0.2, 0.6, 1.0, 0.2, 0.8, 1.0, 1.0, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 0.4, 1.0, 0.4, 0.8, 0.6, 0.8, 1.0, 0.8, 0.2, 0.6, 0.6, 1.0, 1.0, 0.8, 0.8, 0.6, 0.8, 0.8, 1.0, 0.8, 1.0, 1.0, 0.8, 0.8, 0.6, 1.0, 1.0, 0.8, 0.6, 1.0, 0.2, 1.0, 0.8, 0.8, 0.2, 0.4, 1.0, 1.0, 0.6, 0.8, 0.8, 1.0, 0.8, 0.2, 1.0, 0.8, 0.6, 1.0, 0.8, 0.8, 0.8, 1.0, 1.0, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0, 0.8, 1.0, 1.0, 1.0, 0.2, 1.0, 0.8, 0.8, 0.6, 1.0, 1.0, 0.8, 1.0, 1.0, 1.0, 0.4, 0.4, 0.4, 0.4, 0.8, 1.0, 1.0, 1.0, 0.8, 1.0, 1.0, 1.0, 0.8, 0.6, 0.6, 0.6, 0.4, 0.8, 1.0, 0.4, 0.8, 0.8, 0.8, 0.8, 0.6, 0.6, 0.8, 1.0, 0.8, 0.6, 0.6, 0.8, 0.8, 0.8, 0.6, 1.0, 1.0, 0.6, 0.6, 1.0, 0.6, 1.0, 1.0, 1.0, 0.8, 1.0, 0.8, 1.0, 1.0, 0.4, 0.4, 1.0, 1.0, 0.8, 1.0, 0.2]

    # group2 = [0.0, 0.6, 1.0, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 1.0, 0.2, 0.8, 1.0, 0.4, 0.4, 1.0, 0.4, 1.0, 1.0, 0.8, 0.8, 0.4, 0.6, 0.6, 0.4, 1.0, 1.0, 0.6, 0.6, 0.6, 1.0, 0.6, 1.0, 1.0, 0.4, 0.2, 0.8, 1.0, 1.0, 1.0, 1.0, 0.4, 1.0, 0.6, 1.0, 1.0, 0.8, 1.0, 0.8, 0.8, 0.6, 0.6, 0.4, 0.8, 1.0, 1.0, 0.8, 0.2, 0.6, 1.0, 0.6, 0.6, 0.6, 0.8, 1.0, 0.4, 0.8, 0.8, 0.2, 0.8, 0.4, 0.8, 0.6, 0.6, 0.8, 1.0, 0.8, 0.8, 0.2, 0.8, 0.8, 0.8, 0.8, 1.0, 0.8, 1.0, 0.2, 0.6, 1.0, 1.0, 0.4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.4, 0.4, 0.8, 0.6, 0.2, 1.0, 0.8, 1.0, 0.8, 0.8, 1.0, 0.6, 0.6, 0.2, 0.4, 0.4, 0.8, 1.0, 1.0, 0.2, 0.6, 0.6, 0.0, 1.0, 1.0, 0.8, 1.0, 0.8, 1.0, 0.6, 0.6, 0.8, 0.8, 0.8, 0.6, 0.6, 1.0, 1.0, 0.8, 0.8, 0.6, 1.0, 1.0, 0.8, 1.0, 1.0, 0.6, 1.0, 0.8, 0.6, 0.8, 1.0, 0.8, 0.2, 1.0, 0.4]
    group1, group2 = array_pad(group1, group2)

    print(stats.wilcoxon(group1, group2, alternative = "greater").pvalue)
    #  WilcoxonResult(statistic=15.0, pvalue=0.2023283082009374)
    #  Not significantly different

wilcoxon_array()
# wilcoxon_csv()