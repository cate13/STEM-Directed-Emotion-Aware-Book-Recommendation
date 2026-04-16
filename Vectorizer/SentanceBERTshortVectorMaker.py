import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
import numpy as np

class SBERTshortVectorMaker:
    def __init__(self, training_samples, model_name='all-MiniLM-L6-v2', target_dim=40):
        """
        Initializes the model and automatically fits PCA based on training_samples.
        """
        self.model = SentenceTransformer(model_name)
        self.normalizer = Normalizer(norm='l2', copy=False)
        
        # PCA setup
        self.target_dim = target_dim
        self.pca = PCA(n_components=self.target_dim)
        
        # Fit PCA immediately during initialization
        self._initialize_pca(training_samples)

    def _initialize_pca(self, samples):
        if len(samples) < self.target_dim:
            raise ValueError(f"You need at least {self.target_dim} samples to reduce to that dimension.")
        
        # Generate initial high-dim embeddings
        embeddings = self.model.encode(samples)
        
        # Train the PCA to find the 40 most important "directions" in your data
        self.pca.fit(embeddings)

    def get_vector(self, text):
        # 1. Get the original dense embedding (e.g., 384 dims)
        embedding = self.model.encode([text])
        
        # 2. Reduce to 40 dims
        reduced_embedding = self.pca.transform(embedding)
        
        # 3. Re-normalize to unit length
        normalized_embedding = self.normalizer.transform(reduced_embedding)
        
        return normalized_embedding[0]