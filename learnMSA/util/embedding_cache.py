import numpy as np
from typing import Callable, Optional


class EmbeddingCache:
    """ A datastructure to store large amounts of embeddings space efficiently
        in memory by avoiding fragmentation. The embeddings are stored in a
        single chunk of memory and can be accessed by index.
        Stores embeddings in half-precision per default.
    Args:
        seq_lens: An array that contains the lengths of the sequences.
        dim: The dimensionality of the embeddings returned by compute_emb_func.
        dtype: The data type of the embeddings.
        cache: An optional pre-computed cache array of shape
            (sum(seq_lens), dim). If provided, the cache is used directly
            and ``fill_cache`` does not need to be called. The dtype of the
            provided array is preserved.
    """
    def __init__(
        self,
        seq_lens: np.ndarray,
        dim: int,
        dtype: type[np.floating] = np.float16,
        cache: Optional[np.ndarray] = None,
    ) -> None:
        self.seq_lens = seq_lens
        self.dim = dim
        self.cum_lens = np.cumsum(seq_lens)
        if cache is not None:
            expected_rows = self.cum_lens[-1]
            if cache.shape != (expected_rows, dim):
                raise ValueError(
                    f"Expected cache of shape ({expected_rows}, {dim}), "
                    f"got {cache.shape}"
                )
            self.cache = cache
            self._filled = True
        else:
            self.cache = np.zeros((self.cum_lens[-1], dim), dtype=dtype)
            self._filled = False
        self.cum_lens -= seq_lens  # make exclusive cumsum


    def fill_cache(
        self,
        compute_emb_func: Callable[[np.ndarray], np.ndarray],
        batch_size_callback: Callable[[int], int],
        verbose=True,
    ) -> None:
        """ Fill the cache with embeddings.
        Args:
            compute_emb_func: A function that computes the embeddings for a
            batch of sequence indices of the same dtype as the cache.
            batch_size_callback: A function that returns an appropriate batch
            size for a given sequence length.
        """
        L = self.seq_lens
        sorted_indices = np.argsort(-L)
        i = 0
        last = 0
        n = L.size
        # double batch size if half precision for speed
        batch_size_mul = 2 if self.cache.dtype==np.float16 else 1
        while i < n:
            batch_size = batch_size_callback(int(L[sorted_indices[i]]))
            batch_size *= batch_size_mul
            batch_indices = sorted_indices[i:i+batch_size]
            embeddings = compute_emb_func(batch_indices)
            for j,k in enumerate(batch_indices):
                s = self.cum_lens[k]
                self.cache[s:s+L[k]] = embeddings[j, :L[k]]
            i += batch_size
            if verbose:
                for k in range(1,11):
                    if i/n > k/10:
                        if last < k:
                            last = k
                            print(f"{k*10}% done.")
                            break
        self._filled = True


    def get_embedding(self, i: int) -> np.ndarray:
        """ Get the embedding for the i-th sequence in the dataset.
        Args:
            i: The index of the sequence.
        Returns:
            The embedding of the i-th sequence.
        """
        start = self.cum_lens[i]
        return self.cache[start:start+self.seq_lens[i]]


    def is_filled(self) -> bool:
        """ Return True if the fill_cache method has been called and\
            get_embedding can be used, False otherwise.
        """
        return self._filled