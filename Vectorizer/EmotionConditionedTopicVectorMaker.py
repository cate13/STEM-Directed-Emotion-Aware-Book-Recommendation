import numpy as np

class EmotionConditionedTopicVectorMaker():
    def __init__(self, general_emotion_vector = {'Anger': 0.08884270870928422, 'Anticipation': 0.16379956556255712, 'Disgust': 0.06072994596800967, 'Fear': 0.1344441128137556, 'Joy': 0.14161884812313696, 'Sadness': 0.10450661538707133, 'Surprise': 0.07087905991324454, 'Trust': 0.1937157288887748}):
        self.general_emotion_vector = general_emotion_vector

    def get_emotion_conditioned_topic_vector(self, topic_vec, emotion_vec):
        # Convert inputs to arrays
        vec_a = np.array(list(topic_vec.values())) if isinstance(topic_vec, dict) else np.array(topic_vec)
        vec_b1 = np.array(list(emotion_vec.values()))
        vec_b2 = np.array(list(self.general_emotion_vector.values()))

        # 1. Apply Smoothing (Epsilon)
        # Adding 1e-6 ensures that even 0.0 entries have a tiny 'presence'
        vec_a = vec_a + 1e-6
        vec_b1 = vec_b1 + 1e-6

        # 2. Mathematical Interaction
        # Outer product creates a Topic x Emotion matrix
        matrix_m = np.outer(vec_a, vec_b1)
        # Dot product with general emotions collapses it back to a topic-sized vector
        result = np.dot(matrix_m, vec_b2)

        # 3. Normalization
        # This ensures the final vector sums to 1.0, 
        # making the 'superconductor' vector comparable to a 'joyful' one.
        sum_result = np.sum(result)
        if sum_result > 0:
            result = result / sum_result

        return result.tolist()



