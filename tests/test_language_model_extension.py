import numpy as np
from learnMSA.protein_language_models import EmbeddingCache
from learnMSA.msa_hmm import Priors


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
    cache = EmbeddingCache.EmbeddingCache(seq_lens, dim, dtype=np.float32)
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


def test_regularizer() -> None:
    # test the regularizer
    reg_shared = Priors.L2Regularizer(1, 1, True)
    reg_non_shared = Priors.L2Regularizer(1, 1, False)
    reg_shared.build([])
    reg_non_shared.build([])
    #just test the embedding part
    lengths = [5, 6]
    B = np.zeros((2, 20, 101), dtype=np.float32)
    B[0, :2*lengths[0]+2, 25:] = 2.
    B[0, 1:lengths[0]+1, 25:] = 3.
    B[1, :2*lengths[1]+2, 25:] = 5.
    B[1, 1:lengths[1]+1, 25:] = 4.
    r1 = reg_shared.get_l2_loss(B, lengths)
    r2 = reg_non_shared.get_l2_loss(B, lengths)
    assert all(r1[0,:-1] == 75 * 9 + 75 * 4)
    assert r1[0,-1] == 0
    assert all(r1[1,:-1] == 75 * 16 + 75 * 25)
    assert all(r2[0,:-1] == 75 * 9 + 7 * 75 * 4 / 5)
    assert r2[0,-1] == 0
    assert all(r2[1,:-1] == 75 * 16 + 8 * 75 * 25 / 6)
