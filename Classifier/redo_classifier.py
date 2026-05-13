import json
import numpy as np
import pandas as pd
import os
import itertools
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from scipy.stats import wilcoxon
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import PassiveAggressiveClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NON_STEM_PATH = os.path.join(BASE_DIR, "data_exploring", "isbn_files", "750_non_stem_books_read_by_youth_with_descriptions.txt")
STEM_PATH = os.path.join(BASE_DIR, "data_exploring", "isbn_files", "stem_books_read_by_youth.txt")
# VECTOR_FILE_PATH = os.path.join(BASE_DIR, "Classifier", "empath_7d_tf_idf_long_vectors.jsonl")
VECTOR_FILE_PATH = os.path.join(BASE_DIR, "Classifier", "book_vectors_trainging_classifier_include_sbert.jsonl")
VECTOR_SCORES = os.path.join(BASE_DIR, "Classifier", "model_results.csv")



def load_isbns(file_path):
    with open(file_path, 'r') as f:
        # Strip whitespace and ignore empty lines
        return set(line.strip() for line in f if line.strip())

def get_sets():
    non_stem = load_isbns(NON_STEM_PATH)
    stem = load_isbns(STEM_PATH)

    return non_stem, stem

def data_set_up(vector_types=["empath"]):
    non_stem, stem = get_sets()

    data = []
    labels = []

    with open(VECTOR_FILE_PATH, 'r') as f:
        for line in f:
            item = json.loads(line)
            isbn = item['isbn']
            #print(isbn)

            if isbn in non_stem:
                label = 0
            elif isbn in stem:
                label = 1
            else:
                continue
            labels.append(label)

            vec = []
            for vec_type in vector_types:
                if vec_type == "tf_idf" or vec_type == "sentence_bert":
                    vec += item[vec_type]
                else:
                    vec += [item[vec_type][k] for k in sorted(item[vec_type].keys())]
            data.append(vec)

    X = np.array(data)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test

def test_all_models(X_train, X_test, y_train, y_test):
    minmax_scaler = MinMaxScaler()
    X_train_nonneg = minmax_scaler.fit_transform(X_train)
    X_test_nonneg = minmax_scaler.transform(X_test)

    # Standard scaling
    std_scaler = StandardScaler()
    X_train_scaled = std_scaler.fit_transform(X_train)
    X_test_scaled = std_scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    print(f"RandomForestClassifier Accuracy: {rf.score(X_test, y_test):.4f}")

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    print(f"LR Accuracy: {lr.score(X_test_scaled, y_test):.4f}")

    mnb = MultinomialNB()
    mnb.fit(X_train_nonneg, y_train)
    print(f"MultinomialNB Accuracy: {mnb.score(X_test_nonneg, y_test):.4f}")

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    print(f"GaussianNB Accuracy: {gnb.score(X_test, y_test):.4f}")

    clf = SVC(kernel='linear', C=1.0, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"SVC Accuracy: {clf.score(X_test, y_test):.4f}")

    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        eval_metric='logloss'
    )

    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    print(f"k-NN Accuracy: {knn.score(X_test_scaled, y_test):.4f}")

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    print(f"MLP (Neural Net) Accuracy: {mlp.score(X_test_scaled, y_test):.4f}")

    lgbm = LGBMClassifier(random_state=42, verbose=-1)
    lgbm.fit(X_train, y_train)
    print(f"LightGBM Accuracy: {lgbm.score(X_test, y_test):.4f}")

    ada = AdaBoostClassifier(n_estimators=100, random_state=42)
    ada.fit(X_train, y_train)
    print(f"AdaBoost Accuracy: {ada.score(X_test, y_test):.4f}")

    hgb_model = HistGradientBoostingClassifier(random_state=42)
    hgb_model.fit(X_train, y_train)
    print(f"HistGradientBoosting Accuracy: {hgb_model.score(X_test, y_test):.4f}")

    pac = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
    pac.fit(X_train_scaled, y_train)
    print(f"Passive Aggressive Accuracy: {pac.score(X_test_scaled, y_test):.4f}")


def test_all_models_write_to_csv(X_train, X_test, y_train, y_test):
    """Returns a dictionary of scores instead of printing them."""
    results = {}
    
    # Pre-scaling
    minmax_scaler = MinMaxScaler()
    X_train_nonneg = minmax_scaler.fit_transform(X_train)
    X_test_nonneg = minmax_scaler.transform(X_test)

    std_scaler = StandardScaler()
    X_train_scaled = std_scaler.fit_transform(X_train)
    X_test_scaled = std_scaler.transform(X_test)

    # Dictionary of models and which version of X to use
    # Format: { "Column Name": (ModelObject, X_train_to_use, X_test_to_use) }
    models = {
        "RandomForest": (RandomForestClassifier(n_estimators=100, random_state=42), X_train, X_test),
        "LogisticRegression": (LogisticRegression(max_iter=1000), X_train_scaled, X_test_scaled),
        "MultinomialNB": (MultinomialNB(), X_train_nonneg, X_test_nonneg),
        "GaussianNB": (GaussianNB(), X_train, X_test),
        "SVC_Linear": (SVC(kernel='linear', C=1.0, random_state=42), X_train, X_test),
        "XGBoost": (XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'), X_train, X_test),
        "kNN": (KNeighborsClassifier(n_neighbors=5), X_train_scaled, X_test_scaled),
        "MLP_NeuralNet": (MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42), X_train_scaled, X_test_scaled),
        "LightGBM": (LGBMClassifier(random_state=42, verbose=-1), X_train, X_test),
        "AdaBoost": (AdaBoostClassifier(n_estimators=100, random_state=42), X_train, X_test),
        "HistGradientBoost": (HistGradientBoostingClassifier(random_state=42), X_train, X_test),
        "PassiveAggressive": (PassiveAggressiveClassifier(max_iter=1000, random_state=42), X_train_scaled, X_test_scaled)
    }

    for name, (model, xtr, xte) in models.items():
        model.fit(xtr, y_train)
        results[name] = round(model.score(xte, y_test), 4)
    
    return results

vector_options = [
    "empath", 
    "sentence_bert", 
    "tf_idf", 
    "empath_vec_with_base_word_list", 
    "empath_vec_shared_llm_word_lsit"
]

all_results = []

# Generate all possible combinations (from 1 to 5 elements)
for r in range(1, len(vector_options) + 1):
    for combo in itertools.combinations(vector_options, r):
        combo_list = list(combo)
        print(f"Running for: {combo_list}")
        
        # Load and split data
        X_train, X_test, y_train, y_test = data_set_up(combo_list)
        
        # Run models
        scores = test_all_models_write_to_csv(X_train, X_test, y_train, y_test)
        
        # Add metadata for the row
        scores['vector_types'] = ", ".join(combo_list)
        all_results.append(scores)

# Convert to DataFrame and reorder columns so 'vector_types' is first
df = pd.DataFrame(all_results)
cols = ['vector_types'] + [c for c in df.columns if c != 'vector_types']
df = df[cols]

# Save to CSV
df.to_csv(VECTOR_SCORES, index=False)
print("Done! Results saved to model_results.csv")