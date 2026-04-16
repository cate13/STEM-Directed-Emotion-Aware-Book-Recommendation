import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from scipy.spatial.distance import euclidean


# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATH_STEM = os.path.join(
    BASE_DIR, "stem_emotion_relation", "any_age_users_who_like_stem_books_with_vec.jsonl"
)

PATH_NOT_STEM = os.path.join(
    BASE_DIR, "stem_emotion_relation", "any_age_users_who_do_not_like_stem_books_with_vec.jsonl"
)

PATH_MIX = os.path.join(
    BASE_DIR, "stem_emotion_relation", "any_age_users_who_mix_like_stem_books_with_vec.jsonl"
)

def get_lists():
    users_who_like_stem = []
    users_who_do_not_like_stem = []
    users_mix = []

    def process_file(path, target_list):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                user = json.loads(line)
                # Check if key exists AND is not None
                vec = user.get("emotion_vec_all_books")
                if vec is not None:
                    target_list.append(vec)
                else:
                    user_id = user.get("user_id", "Unknown")
                    print(f"Skipping user {user_id}: Vector is null")

    process_file(PATH_STEM, users_who_like_stem)
    process_file(PATH_NOT_STEM, users_who_do_not_like_stem)
    process_file(PATH_MIX, users_mix)

    return users_who_like_stem, users_who_do_not_like_stem, users_mix


emotions = ['Anger', 'Anticipation', 'Disgust', 'Fear',
            'Joy', 'Sadness', 'Surprise', 'Trust']

def dicts_to_matrix(dict_list):
    return np.array([[d[e] for e in emotions] for d in dict_list])


def graph():
    users_who_like_stem, users_who_do_not_like_stem, users_mix = get_lists()
    print(len(users_who_like_stem))
    print(len(users_who_do_not_like_stem))
    print(len(users_mix))

    matrix_stem = dicts_to_matrix(users_who_like_stem)
    matrix_not_stem = dicts_to_matrix(users_who_do_not_like_stem)
    matrix_mix = dicts_to_matrix(users_mix)

    mean_stem = matrix_stem.mean(axis=0)
    mean_not_stem = matrix_not_stem.mean(axis=0)
    mean_mix = matrix_mix.mean(axis=0)

    x = np.arange(len(emotions))
    jitter_strength = 0.04

    plt.figure(figsize=(12,6))

    def plot_group(mat, mean, color, label, offset):
        # jittered scatter
        for i in range(mat.shape[0]):
            jitter = np.random.normal(0, jitter_strength, len(x))
            plt.scatter(x + offset + jitter,
                        mat[i],
                        alpha=0.3,
                        color=color)

        # mean line
        plt.plot(x + offset,
                mean,
                color=color,
                linewidth=3,
                label=label)

    plot_group(matrix_stem, mean_stem, "blue", "Likes STEM", -0.2)
    plot_group(matrix_not_stem, mean_not_stem, "red", "Does Not Like STEM", 0)
    plot_group(matrix_mix, mean_mix, "green", "Mix", 0.2)

    plt.xticks(x, emotions, rotation=45)
    plt.ylabel("Emotion Score")
    plt.title("Emotion Vector Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()


def classify_test():
    users_who_like_stem, users_who_do_not_like_stem, users_mix = get_lists()

    matrix_stem = dicts_to_matrix(users_who_like_stem)
    matrix_not_stem = dicts_to_matrix(users_who_do_not_like_stem)
    matrix_mix = dicts_to_matrix(users_mix)

    X = np.vstack([matrix_stem, matrix_not_stem, matrix_mix])

    y = np.array(
        [0]*len(matrix_stem) +
        [1]*len(matrix_not_stem) +
        [2]*len(matrix_mix)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        stratify=y,
        random_state=42
    )

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print("Test Accuracy:", accuracy)

    scores = cross_val_score(model, X, y, cv=5)
    print("Cross-val accuracy:", scores.mean()) 

    coef_df = pd.DataFrame(
        model.coef_,
        columns=emotions,
        index=["STEM", "Not STEM", "MIX"]
    )

    print(coef_df)

    y_pred = model.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    model_classification(X, y)


def classify_test_just_stem_not_stem():
    users_who_like_stem, users_who_do_not_like_stem, users_mix = get_lists()
    # try and see if adding mix helps
    # users_who_do_not_like_stem.extend(users_mix)

    matrix_stem = dicts_to_matrix(users_who_like_stem)
    matrix_not_stem = dicts_to_matrix(users_who_do_not_like_stem)

    X = np.vstack([matrix_stem, matrix_not_stem])

    y = np.array(
        [0]*len(matrix_stem) +
        [1]*len(matrix_not_stem)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        stratify=y,
        random_state=42
    )

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print("Test Accuracy:", accuracy)

    scores = cross_val_score(model, X, y, cv=5)
    print("Cross-val accuracy:", scores.mean()) 

    coef_df = pd.DataFrame(
        model.coef_,
        columns=emotions,
        index=["Not STEM vs STEM"]
    )

    print(coef_df)

    y_pred = model.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    model_classification(X, y)


def model_classification(X, y):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    plt.scatter(X_2d[y==0,0], X_2d[y==0,1], alpha=0.6, label="List1")
    plt.scatter(X_2d[y==1,0], X_2d[y==1,1], alpha=0.6, label="List2")
    plt.scatter(X_2d[y==2,0], X_2d[y==2,1], alpha=0.6, label="List3")

    plt.legend()
    plt.title("PCA Projection of Emotion Vectors")
    plt.show()

def diagnostic():
    users_who_like_stem, users_who_do_not_like_stem, users_mix = get_lists()

    matrix_stem = dicts_to_matrix(users_who_like_stem)
    matrix_not_stem = dicts_to_matrix(users_who_do_not_like_stem)
    matrix_mix = dicts_to_matrix(users_mix)

    print("Mean vectors:")
    print(matrix_stem.mean(axis=0))
    print(matrix_not_stem.mean(axis=0))
    print(matrix_mix.mean(axis=0))

    print("Distance STEM–NotSTEM:", euclidean(matrix_stem.mean(axis=0), matrix_not_stem.mean(axis=0)))
    print("Distance STEM–MIX:", euclidean(matrix_stem.mean(axis=0), matrix_mix.mean(axis=0)))
    print("Distance NotSTEM–MIX:", euclidean(matrix_not_stem.mean(axis=0), matrix_mix.mean(axis=0)))


def test_each_emotion():
    users_who_like_stem, users_who_do_not_like_stem, users_mix = get_lists()

    matrix_stem = dicts_to_matrix(users_who_like_stem)
    matrix_not_stem = dicts_to_matrix(users_who_do_not_like_stem)

    X = np.vstack([matrix_stem, matrix_not_stem])

    y = np.array(
        [0]*len(matrix_stem) +
        [1]*len(matrix_not_stem)
    )


    for i, emotion in enumerate(emotions):
        X_single = X[:, i].reshape(-1, 1)
        model = LogisticRegression(class_weight='balanced')
        scores = cross_val_score(model, X_single, y, cv=5)
        print(f"{emotion}: {scores.mean():.3f}")

test_each_emotion()