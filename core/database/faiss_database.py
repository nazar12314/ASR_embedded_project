import faiss
import os
import numpy as np

from utils.logger_utils import get_logger


logger = get_logger(__name__)


class FaissDatabase:
    def __init__(
            self,
            index_path: str = 'faiss_index.bin',
            dimension: int = 256
        ):
        self.index_path = index_path
        self.dimension = dimension
        self.faiss_index = self.load_index()

    def load_index(self):
        """
        Load Faiss index from disk or create new one if not exists

        Returns:
            faiss.Index: Faiss index
        """
        if os.path.exists(self.index_path):
            logger.info(f"Loading Faiss index from {self.index_path}")
            return faiss.read_index(self.index_path)
        else:
            logger.info(f"Creating new Faiss index at {self.index_path}")
            return faiss.IndexFlatL2(self.dimension)

    def save_index(self):
        """
        Save Faiss index to disk
        """
        faiss.write_index(self.faiss_index, self.index_path)

    def add_embedding(self, embedding: np.array):
        """
        Add embedding to Faiss index

        Args:
            embedding (np.array): embedding to add to index

        Returns:
            int: index of added embedding
        """
        self.faiss_index.add(np.array([embedding]))

        return self.faiss_index.ntotal - 1

    def search_embedding(self, embedding: np.array, k: int = 1):
        """
        Search for k nearest embeddings in Faiss index

        Args:
            embedding (np.array): embedding to search for
            k (int): number of nearest embeddings to search for

        Returns:
            (np.array, np.array): indices and distances of k nearest embeddings
        """

        return self.faiss_index.search(np.array([embedding]), k)
