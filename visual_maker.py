from Recomender_Helper.vector_helper import graphVector, average_vectors
import json
import random
import matplotlib.pyplot as plt
import numpy as np
from itertools import islice
from tqdm import tqdm

def _load_book_titles():
    print("Loading titles...")
    titles = {}

    with open("processed_data/books_with_subjects.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            isbn = data['ISBN']
            title = data['Book-Title']

            titles[isbn] = title
    
    print("Titles loaded!")
    return titles

BOOK_TITLES = _load_book_titles()

for key, value in islice(BOOK_TITLES.items(), 10):
    print(f"{key}: {value}")

def get_title(isbn):
    return BOOK_TITLES.get(isbn, "Unknown Title")

def load_random_lines(file_path, n=10):
    reservoir = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            # Fill the reservoir with the first 'n' lines
            if i < n:
                reservoir.append(json.loads(line))
            else:
                # For every subsequent line, decide if it should replace an existing one
                j = random.randint(0, i)
                if j < n:
                    reservoir[j] = json.loads(line)
                    
    return reservoir


def plot_emotion_vectors(vec1, vec2, name1, name2, title):
    emotions = vec1.keys()
    
    # Extract values in the same order
    values1 = [vec1.get(emotion, 0) for emotion in emotions]
    values2 = [vec2.get(emotion, 0) for emotion in emotions]
    
    # Set the positions of the bars
    x = np.arange(len(emotions))  
    width = 0.35  # Width of the individual bars

    # Create the plot using subplots
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars
    ax.bar(x - width/2, values1, width, label=name1, color='#3498db') # Blue
    ax.bar(x + width/2, values2, width, label=name2, color='#e67e22') # Orange

    # Add labels and formatting
    ax.set_ylabel('Scores')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(emotions)
    ax.legend()

    # Prevent labels from being cut off
    plt.tight_layout()
    plt.show()

def show_emotion_vec_diff():
    data = load_random_lines("processed_data/book_vectors_base.jsonl", 30)

    for line in data:
        emotion_intensity = line['emotion_intensity']
        emotion = line['emotion']
        isbn = line['isbn']
        title = get_title(isbn)
        if title == "Unknown Title":
            print(isbn)
            continue

        print(emotion)
        print(emotion_intensity)
        plot_emotion_vectors(emotion, emotion_intensity, "NRC-EIL", "EmoLex", title)



def show_average_vec_diff():
    data = load_random_lines("processed_data/book_vectors_base.jsonl", 1000)
    print("got data")

    emotion_intensity_list = []
    emotion_list = []
    for line in data:
        emotion_intensity_list.append(line['emotion_intensity'])
        emotion_list.append(line['emotion'])
    
    average_emotion_intensity_vector = average_vectors(emotion_intensity_list)
    average_emotion_vector = average_vectors(emotion_list)

    print(f"EmoLex: {average_emotion_intensity_vector}")
    print(f"NRC-EIL: {average_emotion_vector}")
    plot_emotion_vectors(average_emotion_vector, average_emotion_intensity_vector, "NRC-EIL", "EmoLex", "Average Emotion Vectors across 1,000 Books")




show_average_vec_diff()