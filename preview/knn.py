import faiss
import numpy as np


class FaissKNeighbors:
    def __init__(self, k=1):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        index = faiss.IndexFlatL2(X.shape[1])
        self.index = index
        self.index.add(X.astype(np.float32))
        self.y = y

    def predict(self, X, k=None):
        if not k:
            k = self.k
        distances, indices = self.index.search(X.astype(np.float32), k=k)
        votes = self.y[indices]
        return votes
