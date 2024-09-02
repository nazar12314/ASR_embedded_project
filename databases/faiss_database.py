import faiss
import os
import numpy as np


class FaissDatabase:
    def __init__(self, index_file='data/faiss_index.bin', dimension=256):
        self.index_file = index_file
        self.dimension = dimension
        self.faiss_index = self.load_index()

    def load_index(self):
        if os.path.exists(self.index_file):
            print(f"Loading Faiss index from {self.index_file}")
            return faiss.read_index(self.index_file)
        else:
            print("Creating a new Faiss index.")
            return faiss.IndexFlatL2(self.dimension)

    def save_index(self):
        faiss.write_index(self.faiss_index, self.index_file)

    def add_embedding(self, embedding):
        self.faiss_index.add(np.array([embedding]))
        return self.faiss_index.ntotal - 1

    def search_embedding(self, embedding, k=1):
        return self.faiss_index.search(np.array([embedding]), k)
