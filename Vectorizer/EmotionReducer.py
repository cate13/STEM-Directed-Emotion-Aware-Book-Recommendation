
class EmotionReducer():
    def __init__(self):
        pass 

    def reduce_emotion_vector(base_emotion_vector):
        combined = {
            "Joy_Sadness": base_emotion_vector.get("Joy", 0) - base_emotion_vector.get("Sadness", 0),
            "Trust_Disgust": base_emotion_vector.get("Trust", 0) - base_emotion_vector.get("Disgust", 0),
            "Fear_Anger": base_emotion_vector.get("Fear", 0) - base_emotion_vector.get("Anger", 0),
            "Surprise_Anticipation": base_emotion_vector.get("Surprise", 0) - base_emotion_vector.get("Anticipation", 0)
        }
        return combined
