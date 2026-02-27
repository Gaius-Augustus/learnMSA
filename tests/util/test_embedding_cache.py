import numpy as np

from learnMSA.util import embedding_cache


def test_embedding_cache() -> None:
    seq_lens = np.array([5, 11, 17, 4, 5])
    dim = 32
    def compute_emb_func(indices):
        batch = np.zeros(
            (indices.size, np.amax(seq_lens[indices]), dim), dtype=np.float32
        )
        for i,j in enumerate(indices):
            batch[i, :seq_lens[j]] = \
                (j+1) * np.ones((seq_lens[j], dim), dtype=np.float32)
        return batch
    # Use float32 to avoid batch size scaling in learnMSA production,
    # float16 (default for dtype) will be used for efficiency and the batch sizes
    # will be automatically increased, which would let this test fail
    cache = embedding_cache.EmbeddingCache(seq_lens, dim, dtype=np.float32)
    num_calls = [0, 0]
    def batch_size_callback(L):
        if L > 10:
            num_calls[0] += 1
            return 1
        else:
            num_calls[1] += 1
            return 2
    assert not cache.is_filled()
    cache.fill_cache(compute_emb_func, batch_size_callback, verbose=False)
    assert cache.is_filled()
    for i in range(len(seq_lens)):
        emb = cache.get_embedding(i)
        np.testing.assert_almost_equal(emb, compute_emb_func(np.array([i]))[0])
    assert num_calls == [2, 2]
    assert np.sum(cache.cache) == np.dot(seq_lens, np.arange(1,len(seq_lens)+1)*dim)


def test_embedding_cache_from_existing() -> None:
    """Test initializing an EmbeddingCache with a pre-computed cache array."""
    seq_lens = np.array([5, 11, 17, 4, 5])
    dim = 32
    # Build the flat cache array that EmbeddingCache would store internally
    rows = []
    for i, L in enumerate(seq_lens):
        rows.append((i + 1) * np.ones((L, dim), dtype=np.float32))
    existing_cache = np.concatenate(rows, axis=0)

    cache = embedding_cache.EmbeddingCache(
        seq_lens, dim, cache=existing_cache
    )
    assert cache.is_filled()
    assert cache.cache.dtype == np.float32
    for i in range(len(seq_lens)):
        expected = (i + 1) * np.ones((seq_lens[i], dim), dtype=np.float32)
        np.testing.assert_array_equal(cache.get_embedding(i), expected)


def test_embedding_cache_from_existing_wrong_shape() -> None:
    """Passing a cache with the wrong shape should raise ValueError."""
    seq_lens = np.array([3, 7])
    dim = 16
    bad_cache = np.zeros((99, dim), dtype=np.float32)
    try:
        embedding_cache.EmbeddingCache(seq_lens, dim, cache=bad_cache)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_embedding_cache_from_existing_preserves_dtype() -> None:
    """The dtype of a provided cache should be preserved, not overridden."""
    seq_lens = np.array([2, 3])
    dim = 4
    existing = np.ones((5, dim), dtype=np.float64)
    cache = embedding_cache.EmbeddingCache(seq_lens, dim, cache=existing)
    assert cache.cache.dtype == np.float64
