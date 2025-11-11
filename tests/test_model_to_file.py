import os
import shutil
import warnings

import numpy as np
import tensorflow as tf

from learnMSA import Configuration
from learnMSA.msa_hmm import Initializers, training
from learnMSA.msa_hmm.AlignmentModel import AlignmentModel
from learnMSA.msa_hmm.Emitter import ProfileHMMEmitter
from learnMSA.msa_hmm.Initializers import ConstantInitializer
from learnMSA.msa_hmm.learnmsa_context import LearnMSAContext
from learnMSA.msa_hmm.model import LearnMSAModel
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset
from learnMSA.msa_hmm.Transitioner import ProfileHMMTransitioner


def string_to_one_hot(s : str) -> tf.Tensor:
    i = [SequenceDataset.alphabet.index(aa) for aa in s]
    return tf.one_hot(i, len(SequenceDataset.alphabet)-1)


def test_model_to_file() -> None:
    # Suppress TensorFlow/Keras library warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="keras")
    warnings.filterwarnings("ignore", category=UserWarning, module="keras")

    test_filepath = "tests/data/test_model"

    # Remove any saved models from previous tests
    shutil.rmtree(test_filepath, ignore_errors=True)
    if os.path.exists(test_filepath+".keras"):
        os.remove(test_filepath+".keras")
    if os.path.exists(test_filepath+".zip"):
        os.remove(test_filepath+".zip")

    # Make a model with some custom parameters to save
    model_len = 10
    custom_transition_init = Initializers.make_default_transition_init(
        MM=7, MI=-5, MD=2, II=5, IM=12, DM=3,
        DD=22, FC=6, FE=7, R=8, RF=-2, T=-10,
    )
    custom_flank_init = ConstantInitializer(2)
    em_init_np = np.random.rand(model_len, len(SequenceDataset.alphabet)-1)
    em_init_np[2:6] = string_to_one_hot("ACGT").numpy()*20.
    custom_emission_init = ConstantInitializer(em_init_np)
    custom_insertion_init = ConstantInitializer(np.random.rand(len(SequenceDataset.alphabet)-1))
    config = Configuration()
    config.training.num_model = 1
    config.training.no_sequence_weights = True
    config.training.length_init = [model_len]

    data = SequenceDataset("tests/data/simple.fa")

    context = LearnMSAContext(config, data)
    context.transitioner = ProfileHMMTransitioner(custom_transition_init, custom_flank_init)
    context.emitter = ProfileHMMEmitter(custom_emission_init, custom_insertion_init)
    encoder_initializer = Initializers.make_default_anc_probs_init(1)
    encoder_initializer[0] = ConstantInitializer(np.random.rand(1, 2))
    context.encoder_initializer = encoder_initializer
    model = LearnMSAModel(context)
    model.compile() #prevents warnings
    model.build()

    #copy current parameter state
    emission_kernel = model.msa_hmm_cell.emitter[0].emission_kernel[0].numpy()
    insertion_kernel = model.msa_hmm_cell.emitter[0].insertion_kernel[0].numpy()
    transition_kernel = {
        key : kernel.numpy()
        for key, kernel in model.msa_hmm_cell.transitioner.transition_kernel[0].items()
    }
    flank_init_kernel = model.msa_hmm_cell.transitioner.flank_init_kernel[0].numpy()
    tau_kernel = model.anc_probs_layer.tau_kernel.numpy()
    seq = np.random.randint(25, size=(1,1,17))
    seq[:,:,-1] = 25
    loglik = model([seq, np.array([[0]])])[1].numpy()

    #make alignment and save
    context.batch_gen.configure(data, config)
    ind = np.array([0,1])
    batch_size = 2
    am = AlignmentModel(data, context.batch_gen, ind, batch_size, model)
    tf.get_logger().setLevel('ERROR') #prints some info and unrelevant warnings
    am.write_models_to_file(test_filepath)
    tf.get_logger().setLevel('WARNING')

    # Remember how the decoded MSA looks and delete the alignment object
    # Todo: the MSA is currently nonsense, but it should be enough to test
    # if Viterbi runs are consistent
    tf.get_logger().setLevel('ERROR')
    # Prints expected warnings about retracing
    msa_str = am.to_string(model_index = 0)
    tf.get_logger().setLevel('WARNING')
    del am

    #load again
    deserialized_am = AlignmentModel.load_models_from_file(
        test_filepath, data, custom_batch_gen=context.batch_gen
    )

    #test if parameters are the same
    deserialized_emission_kernel = deserialized_am.model.msa_hmm_cell.emitter[0].emission_kernel[0].numpy()
    np.testing.assert_equal(emission_kernel, deserialized_emission_kernel)

    deserialized_insertion_kernel = deserialized_am.model.msa_hmm_cell.emitter[0].insertion_kernel[0].numpy()
    np.testing.assert_equal(insertion_kernel, deserialized_insertion_kernel)

    for key, k in transition_kernel.items():
        deserialized_k = deserialized_am.model.msa_hmm_cell.transitioner.transition_kernel[0][key].numpy()
        np.testing.assert_equal(k, deserialized_k)

    deserialized_flank_init_kernel = deserialized_am.model.msa_hmm_cell.transitioner.flank_init_kernel[0].numpy()
    np.testing.assert_equal(flank_init_kernel, deserialized_flank_init_kernel)

    deserialized_tau_kernel = deserialized_am.model.anc_probs_layer.tau_kernel.numpy()
    np.testing.assert_equal(tau_kernel, deserialized_tau_kernel)

    #test if likelihood is the same as before
    loglik_in_deserialized_model = deserialized_am.model([seq, np.array([[0]])])[1].numpy()
    np.testing.assert_equal(loglik, loglik_in_deserialized_model)

    #test MSA as string
    tf.get_logger().setLevel('ERROR') #prints expected warnings about retracing
    msa_str_from_deserialized_model = deserialized_am.to_string(model_index = 0)
    tf.get_logger().setLevel('WARNING')
    assert msa_str == msa_str_from_deserialized_model

    #remove saved models from this test
    shutil.rmtree(test_filepath, ignore_errors=True)
    if os.path.exists(test_filepath+".keras"):
        os.remove(test_filepath+".keras")
    if os.path.exists(test_filepath+".zip"):
        os.remove(test_filepath+".zip")
