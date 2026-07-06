
def normalize(vector):
    # Calculate the total sum of all values
    total_sum = sum(vector.values())

    # Normalize the vector using a dictionary comprehension
    normalized_vector = {key: value / total_sum for key, value in vector.items()}

    # Print the result
    print("Normalized Vector:")
    for key, value in normalized_vector.items():
        print(f"  {key}: {value:.6f}")

    return normalized_vector


# Original vector
vector_12_13 = {
    'Anger': 5.928, 
    'Anticipation': 11.774, 
    'Disgust': 3.710, 
    'Fear': 10.412, 
    'Joy': 15.148, 
    'Sadness': 6.429, 
    'Surprise': 5.140, 
    'Trust': 16.259
}

vector_14_15 = {'Anger': 6.907, 'Anticipation': 10.844, 'Disgust': 4.491, 'Fear': 11.033, 'Joy': 14.164, 'Sadness': 7.528, 'Surprise': 4.784, 'Trust': 15.062}

vector_16_17 = {'Anger': 7.961, 'Anticipation': 12.758, 'Disgust': 4.934, 'Fear': 13.352, 'Joy': 15.926, 'Sadness': 9.764, 'Surprise': 5.474, 'Trust': 16.686}

vector_18_plus = {'Anger': 6.049, 'Anticipation': 10.697, 'Disgust': 3.925, 'Fear': 10.116, 'Joy': 14.021, 'Sadness': 7.400, 'Surprise': 4.871, 'Trust': 16.471}

def get_age_based_emotion_vectors():
    vectors = {}
    vectors[12] = normalize(vector_12_13)
    vectors[13] = normalize(vector_12_13)
    vectors[14] = normalize(vector_14_15)
    vectors[15] = normalize(vector_14_15)
    vectors[16] = normalize(vector_16_17)
    vectors[17] = normalize(vector_16_17)
    vectors[18] = normalize(vector_18_plus)
    vectors[19] = normalize(vector_18_plus)
    return vectors

