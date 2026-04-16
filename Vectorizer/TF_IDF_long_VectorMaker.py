import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

class TF_IDF_long_VectorMaker:
    def __init__(self, all_descriptions, n_components = 40, ):
        self.pipeline = make_pipeline(
            TfidfVectorizer(
                stop_words='english',
                max_features=5000,
                ngram_range=(1, 2)
            )
        )

        self.pipeline.fit(all_descriptions)

    def getTF_IDFvector(self, text):
        return self.pipeline.transform([text]).toarray()[0]