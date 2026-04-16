import pandas as pd

# Load the CSV
df = pd.read_csv("topic_eval_all_models.csv")

# Select all columns except "Topic"
model_columns = df.columns.drop("Topic")

# Filter rows where at least one model column is True
filtered_topics = df[df[model_columns].any(axis=1)]["Topic"]

# Save to txt file (one topic per line)
filtered_topics.to_csv("STEM_topics.txt", index=False, header=False)