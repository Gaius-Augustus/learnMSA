import tensorflow as tf
import numpy as np

class EmbeddingCache:
    """ A datastructure to store large amounts of embeddings space efficiently in memory by avoiding fragmentation.
        The embeddings are stored in a single chunk of memory and can be accessed by index.
    Args:
        seq_lens: An array that contains the lengths of the sequences.
        dim: The dimensionality of the embeddings returned by compute_emb_func.
        compute_emb_func: A function that computes the embeddings for a batch of sequence indices.
        dtype: The data type of the embeddings.
    """
    def __init__(self, seq_lens, dim, compute_emb_func, dtype=np.float32):
        self.seq_lens = seq_lens
        self.dim = dim
        self.compute_emb_func = compute_emb_func
        self.cum_lens = np.cumsum(seq_lens)
        self.cache = np.zeros((self.cum_lens[-1], dim), dtype=dtype)
        self.cum_lens -= seq_lens #make exclusive cumsum


    def fill_cache(self, batch_size_callback):
        """ Fill the cache with embeddings.
        Args:
            batch_size_callback: A function that returns an appropriate batch size for a given sequence length.
        """
        sorted_indices = np.argsort(-self.seq_lens)
        i = 0
        while i < self.seq_lens.size:
            batch_size = batch_size_callback(self.seq_lens[sorted_indices[i]])
            batch_indices = sorted_indices[i:i+batch_size]
            embeddings = self.compute_emb_func(batch_indices)
            for j,k in enumerate(batch_indices):
                start = self.cum_lens[k]
                self.cache[start:start+self.seq_lens[k]] = embeddings[j, :self.seq_lens[k]]
            i += batch_size


    def get_embedding(self, i):
        """ Get the embedding for the i-th sequence in the dataset.
        Args:
            i: The index of the sequence.
        Returns:
            The embedding of the i-th sequence.
        """
        start = self.cum_lens[i]
        return self.cache[start:start+self.seq_lens[i]]