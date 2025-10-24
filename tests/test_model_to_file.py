import os
import shutil
import warnings

import numpy as np
import tensorflow as tf

from learnMSA.msa_hmm import Configuration, Initializers, Training
from learnMSA.msa_hmm.AlignmentModel import AlignmentModel
from learnMSA.msa_hmm.Emitter import ProfileHMMEmitter
from learnMSA.msa_hmm.Initializers import ConstantInitializer
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
    encoder_initializer = Initializers.make_default_anc_probs_init(1)
    encoder_initializer[0] = ConstantInitializer(np.random.rand(1, 2))
    config = Configuration.make_default(1)
    config["transitioner"] = ProfileHMMTransitioner(custom_transition_init, custom_flank_init)
    config["emitter"] = ProfileHMMEmitter(custom_emission_init, custom_insertion_init)
    config["encoder_initializer"] = encoder_initializer
    anc_probs_layer = Training.make_anc_probs_layer(2, config)
    msa_hmm_layer = Training.make_msa_hmm_layer(2, [model_len], config)
    model = Training.generic_model_generator([anc_probs_layer], msa_hmm_layer)
    model.compile() #prevents warnings

    def easy_get_layer(model, name):
        for layer in model.layers:
            if name in layer.name:
                return layer
        return None

    #copy current parameter state
    msa_hmm_layer = easy_get_layer(model, "msa_hmm_layer")
    anc_probs_layer = easy_get_layer(model, "anc_probs_layer")
    emission_kernel = msa_hmm_layer.cell.emitter[0].emission_kernel[0].numpy()
    insertion_kernel = msa_hmm_layer.cell.emitter[0].insertion_kernel[0].numpy()
    transition_kernel = {
        key : kernel.numpy()
        for key, kernel in msa_hmm_layer.cell.transitioner.transition_kernel[0].items()
    }
    flank_init_kernel = msa_hmm_layer.cell.transitioner.flank_init_kernel[0].numpy()
    tau_kernel = anc_probs_layer.tau_kernel.numpy()
    seq = np.random.randint(25, size=(1,1,17))
    seq[:,:,-1] = 25
    loglik = model([seq, np.array([[0]])])[1].numpy()

    #make alignment and save
    with SequenceDataset("tests/data/simple.fa") as data:
        batch_gen = Training.DefaultBatchGenerator()
        batch_gen.configure(data, config)
        ind = np.array([0,1])
        batch_size = 2
        am = AlignmentModel(data, batch_gen, ind, batch_size, model)
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
        deserialized_am = AlignmentModel.load_models_from_file(test_filepath, data, custom_batch_gen=batch_gen)

        #test if parameters are the same
        deserialized_emission_kernel = deserialized_am.msa_hmm_layer.cell.emitter[0].emission_kernel[0].numpy()
        np.testing.assert_equal(emission_kernel, deserialized_emission_kernel)

        deserialized_insertion_kernel = deserialized_am.msa_hmm_layer.cell.emitter[0].insertion_kernel[0].numpy()
        np.testing.assert_equal(insertion_kernel, deserialized_insertion_kernel)

        for key, k in transition_kernel.items():
            deserialized_k = msa_hmm_layer.cell.transitioner.transition_kernel[0][key].numpy()
            np.testing.assert_equal(k, deserialized_k)

        deserialized_flank_init_kernel = deserialized_am.msa_hmm_layer.cell.transitioner.flank_init_kernel[0].numpy()
        np.testing.assert_equal(flank_init_kernel, deserialized_flank_init_kernel)

        deserialized_tau_kernel = easy_get_layer(deserialized_am.model, "anc_probs_layer").tau_kernel.numpy()
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
