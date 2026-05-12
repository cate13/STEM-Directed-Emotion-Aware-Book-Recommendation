import json
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from scipy.stats import wilcoxon

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NON_STEM_PATH = os.path.join(BASE_DIR, "data_exploring", "isbn_files", "750_non_stem_books_read_by_youth_with_descriptions.txt")
STEM_PATH = os.path.join(BASE_DIR, "data_exploring", "isbn_files", "stem_books_read_by_youth.txt")
# VECTOR_FILE_PATH = os.path.join(BASE_DIR, "Classifier", "empath_7d_tf_idf_long_vectors.jsonl")

def load_isbns(file_path):
    with open(file_path, 'r') as f:
        # Strip whitespace and ignore empty lines
        return set(line.strip() for line in f if line.strip())