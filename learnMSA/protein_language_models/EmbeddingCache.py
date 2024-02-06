import numpy as np

class EmbeddingCache:
    """ A datastructure to store large amounts of embeddings space efficiently in memory by avoiding fragmentation.
        The embeddings are stored in a single chunk of memory and can be accessed by index.
    Args:
        seq_lens: An array that contains the lengths of the sequences.
        dim: The dimensionality of the embeddings returned by compute_emb_func.
        dtype: The data type of the embeddings.
    """
    def __init__(self, seq_lens, dim, dtype=np.float32):
        self.seq_lens = seq_lens
        self.dim = dim
        self.cum_lens = np.cumsum(seq_lens)
        self.cache = np.zeros((self.cum_lens[-1], dim), dtype=dtype)
        self.cum_lens -= seq_lens #make exclusive cumsum
        self._filled = False


    def fill_cache(self, compute_emb_func, batch_size_callback, verbose=True):
        """ Fill the cache with embeddings.
        Args:
            compute_emb_func: A function that computes the embeddings for a batch of sequence indices.
            batch_size_callback: A function that returns an appropriate batch size for a given sequence length.
        """
        sorted_indices = np.argsort(-self.seq_lens)
        i = 0
        last = 0
        n = self.seq_lens.size
        while i < n:
            batch_size = batch_size_callback(self.seq_lens[sorted_indices[i]])
            batch_indices = sorted_indices[i:i+batch_size]
            embeddings = compute_emb_func(batch_indices)
            for j,k in enumerate(batch_indices):
                start = self.cum_lens[k]
                self.cache[start:start+self.seq_lens[k]] = embeddings[j, :self.seq_lens[k]]
            i += batch_size
            if verbose:
                for k in range(1,11):
                    if i/n > k/10:
                        if last < k:
                            last = k
                            print(f"{k*10}% done.")
                            break
        self._filled = True


    def get_embedding(self, i):
        """ Get the embedding for the i-th sequence in the dataset.
        Args:
            i: The index of the sequence.
        Returns:
            The embedding of the i-th sequence.
        """
        start = self.cum_lens[i]
        return self.cache[start:start+self.seq_lens[i]]


    def is_filled(self):
        """ Return True if the fill_cache method has been called and get_embedding can be used, False otherwise.
        """
        return self._filled