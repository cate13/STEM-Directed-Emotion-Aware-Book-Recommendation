import pandas as pd
import json
from collections import Counter

def get_top_30_topics(jsonl_file, csv_file):
    # 1. Load the CSV and filter for topics where AtLeast1 is True
    # We handle potential string/boolean conversions for the AtLeast1 column
    df = pd.read_csv(csv_file)
    
    # Ensure AtLeast1 is interpreted as boolean
    if df['AtLeast1'].dtype == object:
        df['AtLeast1'] = df['AtLeast1'].astype(str).str.strip().str.upper() == 'TRUE'
    
    # Create a set of valid topics for fast lookup
    valid_topics = set(df[df['AtLeast1'] == True]['Topic'])

    # 2. Iterate through the JSONL file and count occurrences
    topic_counts = Counter()
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                book = json.loads(line)
                
                # Combine LoC_subjects and Google_categories
                # Use a set to count a topic only once per book
                book_topics = set(book.get('LoC_subjects', [])) | set(book.get('Google_categories', []))
                
                # Update counter for topics present in our valid list
                for topic in book_topics:
                    if topic in valid_topics:
                        topic_counts[topic] += 1
            except json.JSONDecodeError:
                continue

    # 3. Get the top 30 most common topics
    top_30 = topic_counts.most_common(20)
    
    return top_30

# Usage:
results = get_top_30_topics('processed_data/books_with_subjects.jsonl', 'STEM_Books/all_lmm_all_counts.csv')
for topic, count in results:
    print(f"{topic}: {count}")