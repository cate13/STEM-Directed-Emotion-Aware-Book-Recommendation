import json
import os
from collections import Counter

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASSIFIER = os.path.join(
    BASE_DIR, "processed_data", "stem_isbns_from_classifier.txt"
)

COSINE = os.path.join(
    BASE_DIR, "processed_data", "stem_isbns_from_cosine.txt"
)

def compare_isbns(file1_path, file2_path):
    # Read files and strip whitespace/newlines
    with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
        set1 = set(line.strip() for line in f1 if line.strip())
        set2 = set(line.strip() for line in f2 if line.strip())

    # ISBNs in file1 but NOT in file2
    only_in_file1 = set1 - set2
    
    # ISBNs in file2 but NOT in file1
    only_in_file2 = set2 - set1

    return only_in_file1, only_in_file2

# Usage


diff1, diff2 = compare_isbns(CLASSIFIER, COSINE)



print(f"In {CLASSIFIER} but not {COSINE}: {diff1}")
print(f"In {COSINE} but not {CLASSIFIER}: {diff2}")

print(f"{len(diff1)} books in Classifier but not Cosine")
print(f"{len(diff2)} books in Cossine but not Classifier")