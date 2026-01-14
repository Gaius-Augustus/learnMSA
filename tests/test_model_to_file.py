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

    # Load data and set up context
    data = SequenceDataset("tests/data/simple.fa")
    context = LearnMSAContext(config, data)

    model = LearnMSAModel(context)
    model.compile() #prevents warnings
    model.build()

    # Make a snapshot of parameters and likelihood before saving
    weights = [w.numpy() for w in model.trainable_weights]
    seq = np.random.randint(20, size=(1,1,17))
    seq[:,:,-1] = len(SequenceDataset._default_alphabet)-1
    ind = np.array([[0]])
    loglik = model([seq, ind]).numpy()

    #make alignment and save
    context.batch_gen.configure(data, context)
    ind = np.array([0,1])
    batch_size = 2
    am = AlignmentModel(data, context.batch_gen, ind, batch_size, model)
    am.write_models_to_file(test_filepath)

    # # Remember how the decoded MSA looks and delete the alignment object
    # # Todo: the MSA is currently nonsense, but it should be enough to test
    # # if Viterbi runs are consistent
    # tf.get_logger().setLevel('ERROR')
    # # Prints expected warnings about retracing
    # msa_str = am.to_string(model_index = 0)
    # tf.get_logger().setLevel('WARNING')
    # del am

    # #load again
    # deserialized_am = AlignmentModel.load_models_from_file(
    #     test_filepath, data, custom_batch_gen=context.batch_gen
    # )

    # #test if parameters are the same
    # deserialized_emission_kernel = deserialized_am.model.msa_hmm_cell.emitter[0].emission_kernel[0].numpy()
    # np.testing.assert_equal(emission_kernel, deserialized_emission_kernel)

    # deserialized_insertion_kernel = deserialized_am.model.msa_hmm_cell.emitter[0].insertion_kernel[0].numpy()
    # np.testing.assert_equal(insertion_kernel, deserialized_insertion_kernel)

    # for key, k in transition_kernel.items():
    #     deserialized_k = deserialized_am.model.msa_hmm_cell.transitioner.transition_kernel[0][key].numpy()
    #     np.testing.assert_equal(k, deserialized_k)

    # deserialized_flank_init_kernel = deserialized_am.model.msa_hmm_cell.transitioner.flank_init_kernel[0].numpy()
    # np.testing.assert_equal(flank_init_kernel, deserialized_flank_init_kernel)

    # deserialized_tau_kernel = deserialized_am.model.anc_probs_layer.tau_kernel.numpy()
    # np.testing.assert_equal(tau_kernel, deserialized_tau_kernel)

    # #test if likelihood is the same as before
    # loglik_in_deserialized_model = deserialized_am.model([seq, np.array([[0]])])[1].numpy()
    # np.testing.assert_equal(loglik, loglik_in_deserialized_model)

    # #test MSA as string
    # tf.get_logger().setLevel('ERROR') #prints expected warnings about retracing
    # msa_str_from_deserialized_model = deserialized_am.to_string(model_index = 0)
    # tf.get_logger().setLevel('WARNING')
    # assert msa_str == msa_str_from_deserialized_model

    # #remove saved models from this test
    # shutil.rmtree(test_filepath, ignore_errors=True)
    # if os.path.exists(test_filepath+".keras"):
    #     os.remove(test_filepath+".keras")
    # if os.path.exists(test_filepath+".zip"):
    #     os.remove(test_filepath+".zip")
