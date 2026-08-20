from itertools import combinations
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa
from sklearn.metrics import cohen_kappa_score

df = pd.read_csv("STEM_Annontate/Annotated Books - 400.csv")

users = [col for col in df.columns if col.startswith("user_")]

users = [user for user in users if df[user].notna().any()]

print("Users with ratings:", users)

# --------------------------------------------------
# Pairwise Cohen's kappa
# --------------------------------------------------
cohen_results = {}

for u1, u2 in combinations(users, 2):
    # Drop rows where either rater in the pair has a missing value
    pair_df = df[[u1, u2]].dropna()

    # Only calculate if there are ratings to compare
    if len(pair_df) > 0:
        score = cohen_kappa_score(
            pair_df[u1],
            pair_df[u2]
        )
        cohen_results[f"{u1} vs {u2}"] = score

# --------------------------------------------------
# Fleiss' kappa
# --------------------------------------------------
fleiss_df = df[users].dropna()

# Create category matrix of counts for each subject
rating_counts = pd.DataFrame(
    {
        "STEM": (fleiss_df == "STEM").sum(axis=1),
        "Not STEM": (fleiss_df == "Not STEM").sum(axis=1),
    }
)

fleiss = fleiss_kappa(rating_counts.to_numpy())

# --------------------------------------------------
# Output
# --------------------------------------------------
print("=== Cohen's Kappa (Pairwise) ===")
for pair, score in cohen_results.items():
    print(f"{pair}: {score:.4f}")

print("\n=== Fleiss' Kappa (Overall) ===")
print(f"Fleiss' kappa: {fleiss:.4f}")