import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
import re

class W2VVectorMaker:
    def __init__(self, all_descriptions, vector_size=100, window=5, min_count=2):
        # 1. Tokenize the input text (Word2Vec needs lists of words)
        tokenized_data = [self._tokenize(doc) for doc in all_descriptions]
        
        # 2. Train the Word2Vec model
        self.model = Word2Vec(
            sentences=tokenized_data, 
            vector_size=vector_size, 
            window=window, 
            min_count=min_count, 
            workers=4
        )
        # Precompute L2-normalized vectors for better performance
        self.wv = self.model.wv 

    def _tokenize(self, text):
        # Simple cleanup: lowercase and remove non-alphanumeric characters
        return re.findall(r'\w+', text.lower())

    def get_vector(self, text):
        tokens = self._tokenize(text)
        # Extract vectors for words that actually exist in our vocabulary
        vectors = [self.wv[word] for word in tokens if word in self.wv]

        if not vectors:
            # Return zero vector if no words from the text are in the vocabulary
            return np.zeros(self.model.vector_size).tolist()

        # 3. Aggregate: Take the mean of all word vectors
        mean_vector = np.mean(vectors, axis=0)
        
        # 4. Normalize (to match your previous pipeline's behavior)
        vector = normalize(mean_vector.reshape(1, -1))[0]
        return vector.tolist()