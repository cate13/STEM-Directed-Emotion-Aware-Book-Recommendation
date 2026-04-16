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
VECTOR_FILE_PATH = os.path.join(BASE_DIR, "Classifier", "empath_7d_sentence_bert_long_vectors.jsonl")

def load_isbns(file_path):
    with open(file_path, 'r') as f:
        # Strip whitespace and ignore empty lines
        return set(line.strip() for line in f if line.strip())
    

def get_sets():
    non_stem = load_isbns(NON_STEM_PATH)
    stem = load_isbns(STEM_PATH)

    return non_stem, stem

def data_set_up():
    non_stem, stem = get_sets()

    data = []
    labels = []

    with open(VECTOR_FILE_PATH, 'r') as f:
        for line in f:
            item = json.loads(line)
            isbn = item['isbn']

            if isbn in non_stem:
                label = 0
            elif isbn in stem:
                label = 1
            else:
                continue
    
            empath_vec = [item['empath'][k] for k in sorted(item['empath'].keys())]
            sentence_bert_vec = item['sentence_bert']
            
            full_feature_vector = empath_vec + sentence_bert_vec
            
            data.append(full_feature_vector)
            labels.append(label)

    X = np.array(data)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test

def just_x_y():
    non_stem, stem = get_sets()

    data = []
    labels = []

    with open(VECTOR_FILE_PATH, 'r') as f:
        for line in f:
            item = json.loads(line)
            isbn = item['isbn']

            if isbn in non_stem:
                label = 0
            elif isbn in stem:
                label = 1
            else:
                continue
    
            empath_vec = [item['empath'][k] for k in sorted(item['empath'].keys())]
            sentence_bert_vec = item['sentence_bert']
            
            full_feature_vector = empath_vec + sentence_bert_vec
            
            data.append(full_feature_vector)
            labels.append(label)

    X = np.array(data)
    y = np.array(labels)
    return X, y


def try_RandomForestClassifier(X_train, X_test, y_train, y_test ):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def try_LogisticRegression_and_MultinomialNB(X_train, X_test, y_train, y_test ):
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)

    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    print(f"Logistic Reg Accuracy: {lr_model.score(X_test, y_test):.4f}")
    print(f"Naive Bayes Accuracy: {nb_model.score(X_test, y_test):.4f}")

def try_SVM(X_train, X_test, y_train, y_test):
    clf = SVC(kernel='linear', C=1.0, random_state=42)
    
    print("Training Linear SVM (No Scaling)...")
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print(f"SVM Accuracy: {clf.score(X_test, y_test):.4f}")
    print(classification_report(y_test, y_pred))

def try_XGBoost(X_train, X_test, y_train, y_test):
    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    print("Training XGBoost...")
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)

    print("\n--- XGBoost Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def try_LogisticRegression_and_GaussianNB(X_train, X_test, y_train, y_test):
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)

    # Use Gaussian instead of Multinomial
    nb_model = GaussianNB() 
    nb_model.fit(X_train, y_train)

    print(f"Logistic Reg Accuracy: {lr_model.score(X_test, y_test):.4f}")
    print(f"Gaussian NB Accuracy: {nb_model.score(X_test, y_test):.4f}")

def try_MultinomialNB_with_MINMAX(X_train, X_test, y_train, y_test ):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    nb_model = MultinomialNB()
    nb_model.fit(X_train_scaled, y_train)
    print(f"Naive Bayes Accuracy: {nb_model.score(X_test_scaled, y_test):.4f}")
    return nb_model, scaler

def classify_new_books(model, scaler, threshold=0.8, new_jsonl_path="Classifier/books_for_stem_classification.jsonl"):
    """
    Returns only the ISBNs that the model is highly confident are STEM.
    """
    isbns = []
    features = []

    # 1. Load data
    with open(new_jsonl_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            empath_vec = [item['empath_7D'][k] for k in sorted(item['empath_7D'].keys())]
            sentence_bert_vec = item['SBERT']
            
            isbns.append(item['isbn'])
            features.append(empath_vec + sentence_bert_vec)

    # 2. Scale and Predict Probabilities
    X_new_scaled = scaler.transform(np.array(features))
    # probs is an array of [prob_non_stem, prob_stem]
    probs = model.predict_proba(X_new_scaled)
    
    # 3. Filter for High Confidence
    verified_stem_isbns = []
    for i, prob_stem in enumerate(probs[:, 1]):
        if prob_stem >= threshold:
            verified_stem_isbns.append(isbns[i])
            
    return verified_stem_isbns


def all_model_accuracy(X_train, X_test, y_train, y_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    nb_model = MultinomialNB()
    nb_model.fit(X_train_scaled, y_train)
    print(f"Naive Bayes Accuracy: {nb_model.score(X_test_scaled, y_test):.4f}")

    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)

    nb_model = GaussianNB() 
    nb_model.fit(X_train, y_train)

    print(f"Logistic Reg Accuracy: {lr_model.score(X_test, y_test):.4f}")
    print(f"Gaussian NB Accuracy: {nb_model.score(X_test, y_test):.4f}")

    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)

    print(f"XGBClassifier Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    clf = SVC(kernel='linear', C=1.0, random_state=42)

    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print(f"SVC Accuracy: {clf.score(X_test, y_test):.4f}")

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(f"RandomForestClassifier Accuracy: {accuracy_score(y_test, y_pred):.4f}")



def compare_to_multinomial(X, y):
    # 1. Setup Data: MNB needs scaled, non-negative data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Define Models
    # Note: We use a dictionary to loop through them easily
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "GaussianNB": GaussianNB(),
        "XGBClassifier": XGBClassifier(eval_metric='logloss'),
        "SVC": SVC(kernel='linear'),
        "RandomForest": RandomForestClassifier(n_estimators=100)
    }
    
    mnb = MultinomialNB()
    
    # 3. Setup Cross-Validation (10 folds, 3 repeats = 30 samples)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
    
    # 4. Get Benchmark Scores for MultinomialNB
    print("Computing MultinomialNB baseline...")
    mnb_scores = cross_val_score(mnb, X_scaled, y, cv=cv, scoring='accuracy')
    
    results = []

    # 5. Compare each model to MNB
    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        # Use scaled data for all to ensure a fair fight (LogReg and SVC need it!)
        current_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        
        # Perform Wilcoxon Signed-Rank Test
        stat, p_val = wilcoxon(mnb_scores, current_scores)
        
        # Determine significance (Alpha = 0.05)
        is_significant = "Yes" if p_val < 0.05 else "No"
        better_model = "MultinomialNB" if mnb_scores.mean() > current_scores.mean() else name
        
        results.append({
            "Model": name,
            "MNB Mean Acc": f"{mnb_scores.mean():.4f}",
            "Model Mean Acc": f"{current_scores.mean():.4f}",
            "P-Value": f"{p_val:.9f}",
            "Significant?": is_significant,
            "Winner": better_model if is_significant == "Yes" else "Tie/Inconclusive"
        })

    return pd.DataFrame(results)


def save_stem_isbns(results):
    with open("processed_data/stem_isbns_from_classifier.txt", 'w', encoding='utf-8') as f1:
        for isbn in results:
            f1.write(isbn + "\n")



X, y = just_x_y()

print(compare_to_multinomial(X, y))

# X_train, X_test, y_train, y_test = data_set_up()

# all_model_accuracy(X_train, X_test, y_train, y_test)

# model, scaler = try_MultinomialNB_with_MINMAX(X_train, X_test, y_train, y_test)

# new_results = classify_new_books(model, scaler)

# save_stem_isbns(new_results)