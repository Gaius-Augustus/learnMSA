import os
import shutil
import warnings

import numpy as np
import tensorflow as tf

from learnMSA import Configuration
from learnMSA.msa_hmm import Initializers, training
from learnMSA.msa_hmm.alignment_model import AlignmentModel
from learnMSA.msa_hmm.Emitter import ProfileHMMEmitter
from learnMSA.msa_hmm.Initializers import ConstantInitializer
from learnMSA.msa_hmm.learnmsa_context import LearnMSAContext
from learnMSA.model.model import LearnMSAModel
from learnMSA.util.sequence_dataset import SequenceDataset
from learnMSA.msa_hmm.Transitioner import ProfileHMMTransitioner


def string_to_one_hot(s : str) -> tf.Tensor:
    i = [SequenceDataset._default_alphabet.index(aa) for aa in s]
    return tf.one_hot(i, len(SequenceDataset._default_alphabet)-1)


def test_model_to_file() -> None:
    test_filepath = "tests/data/test_model"

    # Remove any saved models from previous tests
    shutil.rmtree(test_filepath, ignore_errors=True)
    if os.path.exists(test_filepath+".keras"):
        os.remove(test_filepath+".keras")
    if os.path.exists(test_filepath+".zip"):
        os.remove(test_filepath+".zip")

    # Make a model with some custom parameters to save
    model_len = 10
    em_init = np.random.rand(model_len, len(SequenceDataset._default_alphabet)-1)
    em_init[2:6] = string_to_one_hot("ACGT").numpy()*20.
    ins_init = np.random.rand(len(SequenceDataset._default_alphabet)-1)
    ins_init /= np.sum(ins_init)

    # Set up configuration
    config = Configuration()
    config.training.num_model = 1
    config.training.no_sequence_weights = True
    config.training.length_init = [model_len]
    config.hmm.match_emissions = em_init
    config.hmm.insert_emissions = ins_init
    config.hmm.p_match_match = 0.65
    config.hmm.p_match_insert = 0.15
    config.hmm.p_match_end = 0.12
    config.hmm.p_insert_insert = 0.45
    config.hmm.p_delete_delete = 0.42
    config.hmm.p_left_left = 0.68
    config.hmm.p_right_right = 0.68
    config.hmm.p_unannot_unannot = 0.68
    config.hmm.p_end_unannot = 1e-4
    config.hmm.p_end_right = 0.48
    config.hmm.use_prior_for_emission_init = False

    # Load data and set up context
    data = SequenceDataset("tests/data/simple.fa")
    context = LearnMSAContext(config, data)

    model = LearnMSAModel(context)
    model.compile()
    model.build()

    # Make a snapshot of parameters and likelihood before saving
    weights = [w.numpy() for w in model.trainable_weights]
    seq = np.random.randint(20, size=(1,1,17))
    seq[:,:,-1] = len(SequenceDataset._default_alphabet)-1
    loglik = model([seq, np.array([[0]])]).numpy()

    #make alignment and save
    context.batch_gen.configure(data, context)
    ind = np.array([0,1])
    am = AlignmentModel(data, model, ind)
    am.save(test_filepath)

    # Decode MSA to string and delete model from memory
    # TODO: the MSA is currently nonsense, but it should be enough to test
    # if Viterbi runs are consistent
    msa_str = am.to_string(model_index = 0)
    del am

    #load again
    am2 = AlignmentModel.load(test_filepath, data)

    # Assert that truly a new model was loaded
    assert am2.model is not model

    # Test if parameters are the same
    deserialized_weights = [w.numpy() for w in am2.model.trainable_weights]
    assert len(weights) == len(deserialized_weights)
    for w, dw in zip(weights, deserialized_weights):
        np.testing.assert_allclose(w, dw, rtol=1e-6, atol=1e-6)

    # Test if likelihood is the same as before
    loglik_in_deserialized_model = am2.model([seq, np.array([[0]])]).numpy()
    np.testing.assert_allclose(
        loglik, loglik_in_deserialized_model, rtol=1e-6, atol=1e-6
    )

    # Test if the decoded MSA is the same
    msa_str_from_deserialized_model = am2.to_string(model_index=0)
    assert msa_str == msa_str_from_deserialized_model

    # Clean up: remove saved models from this test
    shutil.rmtree(test_filepath, ignore_errors=True)
    if os.path.exists(test_filepath+".keras"):
        os.remove(test_filepath+".keras")
    if os.path.exists(test_filepath+".zip"):
        os.remove(test_filepath+".zip")
