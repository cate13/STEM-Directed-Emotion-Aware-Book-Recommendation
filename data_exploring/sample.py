import json
import os
import csv
import random

random.seed(30)

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_JSONL = os.path.join(
    BASE_DIR, "processed_data", "books_with_subjects_read_by_younger_readers.jsonl"
)

OUTPUT_JSONL = os.path.join(
    BASE_DIR, "data_exploring", "sample_400.jsonl"
)

OUTPUT_CSV = os.path.join(
    BASE_DIR, "data_exploring", "sample_400.csv"
)

SAMPLE_SIZE = 400

# Read all records
records = []
with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

# Check there are enough records
if len(records) < SAMPLE_SIZE:
    raise ValueError(
        f"Input file only contains {len(records)} records, "
        f"but {SAMPLE_SIZE} were requested."
    )

# Randomly sample without replacement
sampled_records = random.sample(records, SAMPLE_SIZE)

# Write the sampled JSONL
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for record in sampled_records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# Create the CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["isbn", "user 1", "user 2"])

    for record in sampled_records:
        writer.writerow([record["ISBN"], "", ""])

print(f"Wrote {OUTPUT_JSONL}")
print(f"Wrote {OUTPUT_CSV}")