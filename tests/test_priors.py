import numpy as np
import tensorflow as tf

from learnMSA.msa_hmm import Priors


def test_amino_acid_match_prior() -> None:
    prior = Priors.AminoAcidPrior(dtype=tf.float64)
    prior.build([])
    model_lengths = [2, 5, 3]
    num_models = len(model_lengths)
    max_len = max(model_lengths)
    max_num_states = 2 * max_len + 3
    B = np.random.rand(3, max_num_states, 26)
    B /= np.sum(B, -1, keepdims=True)
    pdf = prior(B, lengths=model_lengths)
    assert pdf.shape == (num_models, max_len)
    for i, l in enumerate(model_lengths):
        np.testing.assert_equal(pdf[i, l:].numpy(), 0.)
