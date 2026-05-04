import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_PATH = os.path.join(
    BASE_DIR, "processed_data", "book_vectors_base.jsonl"
)

def reduce_emotion_vector(base_emotion_vector):
    combined = {
        "Joy_Sadness": base_emotion_vector.get("Joy", 0) - base_emotion_vector.get("Sadness", 0),
        "Trust_Disgust": base_emotion_vector.get("Trust", 0) - base_emotion_vector.get("Disgust", 0),
        "Fear_Anger": base_emotion_vector.get("Fear", 0) - base_emotion_vector.get("Anger", 0),
        "Surprise_Anticipation": base_emotion_vector.get("Surprise", 0) - base_emotion_vector.get("Anticipation", 0)
    }
    return combined

data = []
with open(VECTOR_PATH, 'r') as f:
    for line in f:
        item = json.loads(line)
        
        # Get STEM values
        # stem_vals = item.get('empath', {})
        tfidf_list = item.get('tf_idf', [])
        tfidf_vals = {f"tfidf_{i}": val for i, val in enumerate(tfidf_list)}
        
        # Reduce Emotions
        #reduced_emotions = reduce_emotion_vector(item.get('emotion', {}))
        emotions = item.get('emotion_intensity', [])
        
        # Merge into a single flat dictionary for the row
        #row = {**stem_vals, **reduced_emotions}
        row = {**tfidf_vals, **emotions}
        data.append(row)

df = pd.DataFrame(data)

# 2. Define our column groups
# stem_cols = ['science', 'technology', 'engineering', 'mathematics']
stem_cols = [f"tfidf_{i}" for i in range(40)]
# emotion_cols = ['Joy_Sadness', 'Trust_Disgust', 'Fear_Anger', 'Surprise_Anticipation']
emotion_cols = ['Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust']

# 3. Calculate Correlation Matrix
# We only want the correlation BETWEEN these two groups, not within them
corr_matrix = df.corr().loc[stem_cols, emotion_cols]

# 4. Visualize
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title("STEM Topic vs. Emotion Pair Correlation")
plt.show()

# 5. Export for flavoring use
print(corr_matrix.to_dict())