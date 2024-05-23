import faiss
import numpy as np


class FaissKNeighbors:
    def __init__(self, k=1):
        self.index = None
        self.y = None
        self.k = k
        self.res = faiss.StandardGpuResources()

    def fit(self, X, y):
        index = faiss.IndexFlatL2(X.shape[1])
        self.index = faiss.index_cpu_to_all_gpus(index)
        self.index.add(X.astype(np.float32))
        self.y = y

    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        votes = self.y[indices]
        return votes
