import sys 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#do not print tf info/warnings on import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import unittest
import numpy as np
import tensorflow as tf
#revert back to default and set the logger level individually per test case
#globally omitting all warnings for the entire test suite should be avoided
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
tf.get_logger().setLevel('WARNING')
from learnMSA.msa_hmm import Align, Emitter, Transitioner, Initializers, MsaHmmCell, MsaHmmLayer, Training, Configuration, Viterbi, AncProbsLayer, Priors, DirichletMixture, Utility
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset, AlignedDataset
from learnMSA.msa_hmm.AlignmentModel import AlignmentModel, non_homogeneous_mask_func, find_faulty_sequences
from learnMSA.protein_language_models import Common, DataPipeline, TrainingUtil, MvnMixture, EmbeddingCache
import itertools
import shutil
from test import RefModels as ref
from test import TestSupervisedTraining

class TestDataset(unittest.TestCase):

    def test_records(self):
        for ind in [True, False]:
            with SequenceDataset("test/data/egf.fasta", "fasta", indexed=ind) as data:
                self.assertEqual(data.indexed, ind)
                get_record = lambda i: (str(data.get_record(i).seq).replace('.', '').upper(), str(data.get_record(i).id))
                self.assertEqual(data.num_seq, 7774)
                self.assertEqual(get_record(0), ("CDPNPCYNHGTCSLRATGYTCSCLPRYTGEH", "B3RNP9_TRIAD/78-108"))
                self.assertEqual(get_record(9), ("NACDRVRCQNGGTCQLKTLEDYTCSCANGYTGDH", "B3N1W3_DROAN/140-173"))
                self.assertEqual(get_record(27), ("CNNPCDASPCLNGGTCVPVNAQNYTCTCTNDYSGQN", "B3RNP6_TRIAD/203-238"))
                self.assertEqual(get_record(-1), ("TASCQDMSCSKQGECLETIGNYTCSCYPGFYGPECEYVRE", "1fsb"))
            
            with SequenceDataset("test/data/PF00008_uniprot.fasta", "fasta") as data:
                get_record = lambda i: str(data.get_record(i).seq).replace('.', '').upper()
                self.assertEqual(get_record(0), "PSPCQNGGLCFMSGDDTDYTCACPTGFSG")
                self.assertEqual(get_record(7), "SSPCQNGGMCFMSGDDTDYTCACPTGFSG")
                self.assertEqual(get_record(-1), "CSSSPCNAEGTVRCEDKKGDFLCHCFTGWAGAR")


    def test_encoding(self):
        for ind in [True, False]:
            with SequenceDataset("test/data/felix.fa", "fasta", indexed=ind) as data:
                np.testing.assert_equal(data.get_encoded_seq(0), [13, 6, 10, 9, 11])

            
    def test_ambiguous_amino_acids(self):
        for ind in [True, False]:
            with SequenceDataset("test/data/ambiguous.fasta", "fasta", indexed=ind) as data:
                # seq as string
                self.assertEqual(data.get_record(0).seq, "AGCTBZJbzj")
                # encoded
                np.testing.assert_equal(data.get_encoded_seq(0), [0, 7, 4, 16, 20, 20, 20, 20, 20, 20])


    def test_remove_gaps(self):
        for ind in [True, False]:
            with SequenceDataset("test/data/egf.ref", "fasta", indexed=ind) as data:
                ref = "GTSHLVKCAEKEKTFCVNGGECFMVKDLSNPSRYLCKCQPGFTG----ARCTENVPMKVQNQEKAEELYQK"
                np.testing.assert_equal(str(data.get_record(5).seq), ref)
                np.testing.assert_equal(data.get_encoded_seq(5), [SequenceDataset.alphabet.index(a) for a in ref.replace('-', '')])
                np.testing.assert_equal(data.get_encoded_seq(5, remove_gaps=False), [SequenceDataset.alphabet.index(a) for a in ref])


    def test_invalid_symbol(self):
        for ind in [True, False]:
            with SequenceDataset("test/data/unknown_symbol.fasta", "fasta", indexed=ind) as data:
                self.assertEqual(str(data.get_record(0).seq), "AGTCGTA?GTCGTAAGTCG????TAAGTCGTAAGTCGTA")
                invalid = False
                try:
                    data.get_encoded_seq(0)
                except ValueError:
                    invalid = True
                self.assertTrue(invalid)


    def test_invalid_format(self):
        for test_file in ["faulty_format", "single_sequence", "empty_sequence", "empty_seqid"]:
            invalid = False
            with SequenceDataset(f"test/data/{test_file}.fasta", "fasta", indexed=False) as data:
                try:
                    data.validate_dataset()
                except ValueError:
                    invalid = True
                self.assertTrue(invalid, test_file)


    def test_aligned_dataset(self):
        for ind in [True, False]:
            with AlignedDataset("test/data/felix_msa.fa", "fasta", indexed=ind) as data:
                self.assertEqual(data.alignment_len, 8)
                np.testing.assert_equal(data.seq_lens, [5, 8, 5])
                np.testing.assert_equal(data.starting_pos, [0, 5, 13])
                np.testing.assert_equal(data.get_column_map(0), [3,4,5,6,7])
                np.testing.assert_equal(data.get_column_map(1), [0,1,2,3,4,5,6,7])
                np.testing.assert_equal(data.get_column_map(2), [1,2,3,4,7])


    def test_invalid_msa(self):
        invalid = False
        try:
            AlignedDataset("test/data/faulty_msa.fasta", "fasta", indexed=False)
        except ValueError:
            invalid = True
        self.assertTrue(invalid)


    def test_from_sequences(self):
        sequences = [("seq1", "FELIX"), ("seq2", "FEIX")]
        with SequenceDataset(sequences=sequences) as data:
            self.assertEqual(data.num_seq, 2)
            np.testing.assert_equal(data.get_encoded_seq(0), [13, 6, 10, 9, 20])
            np.testing.assert_equal(data.get_encoded_seq(1), [13, 6, 9, 20])


    def test_from_alignment(self):
        sequences = [("seq1", "FELIX"), ("seq2", "FE-IX")]
        with AlignedDataset(aligned_sequences=sequences) as data:
            self.assertEqual(data.num_seq, 2)
            np.testing.assert_equal(data.get_encoded_seq(0), [13, 6, 10, 9, 20])
            np.testing.assert_equal(data.get_encoded_seq(1), [13, 6, 9, 20])
            np.testing.assert_equal(data.get_column_map(0), [0,1,2,3,4])
            np.testing.assert_equal(data.get_column_map(1), [0,1,3,4])


    def test_file_output_formats(self):
        #write an alignment to various formats
        for fmt in ["fasta", "clustal", "stockholm"]:
            with AlignedDataset(aligned_sequences=[("seq1", "FELIX"), ("seq2", "FE-IX"), ("seq3", "-ELI-")]) as data:
                data.write("example."+fmt, fmt)
        #read it back in and check if it is the same
        for fmt in ["fasta", "clustal", "stockholm"]:
            with AlignedDataset("example."+fmt, fmt) as data:
                self.assertEqual(data.num_seq, 3)
                np.testing.assert_equal(data.get_encoded_seq(0), [13, 6, 10, 9, 20])
                np.testing.assert_equal(data.get_encoded_seq(1), [13, 6, 9, 20])
                np.testing.assert_equal(data.get_encoded_seq(2), [6, 10, 9])
                np.testing.assert_equal(data.get_column_map(0), [0,1,2,3,4])
                np.testing.assert_equal(data.get_column_map(1), [0,1,3,4])
                np.testing.assert_equal(data.get_column_map(2), [1,2,3])



class TestMsaHmmCell(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super(TestMsaHmmCell, self).__init__(*args, **kwargs)
        self.length = [4,3]
        self.emission_init = [ref.make_emission_init_A(), ref.make_emission_init_B()]
        self.insertion_init = [ref.make_insertion_init() for i in range(2)]
        self.transition_init = [ref.make_transition_init_A(), ref.make_transition_init_B()]
        A1, B1, I1 = ref.get_ref_model_A()
        A2, B2, I2 = ref.get_ref_model_B()
        self.A_ref = [A1, A2]
        self.B_ref = [B1, B2]
        self.init_ref = [I1, I2]
        self.ref_alpha = [ref.get_ref_forward_A(), ref.get_ref_forward_B()]
        self.ref_beta = [ref.get_ref_backward_A(), ref.get_ref_backward_B()]
        self.ref_lik = [ref.get_ref_lik_A(), ref.get_ref_lik_B()]
        self.ref_scaled_alpha = [ref.get_ref_scaled_forward_A(), ref.get_ref_scaled_forward_B()]
        self.ref_posterior_probs = [ref.get_ref_posterior_probs_A(), ref.get_ref_posterior_probs_B()]
        self.ref_gamma = [ref.get_ref_viterbi_variables_A(), ref.get_ref_viterbi_variables_B()]
        self.ref_viterbi_path = [ref.get_ref_viterbi_path_A(), ref.get_ref_viterbi_path_B()]
    
    def make_test_cell(self, models):
        if not hasattr(models, '__iter__'):
            models = [models]
        length = [self.length[i] for i in models]
        e = [self.emission_init[i] for i in models]
        i = [self.insertion_init[i] for i in models]
        emitter = Emitter.ProfileHMMEmitter(emission_init = e, insertion_init = i)
        t = [self.transition_init[i] for i in models]
        f = [Initializers.make_default_flank_init() for _ in models]
        transitioner = Transitioner.ProfileHMMTransitioner(transition_init = t, flank_init = f)
        hmm_cell = MsaHmmCell.MsaHmmCell(length, dim=3, emitter=emitter, 
                                        transitioner=transitioner, use_step_counter=True)
        hmm_cell.build((None,None,3))
        return hmm_cell, length
   
    def test_single_models(self):
        for i in range(2):
            hmm_cell, length = self.make_test_cell(i)
            hmm_cell.recurrent_init()
            A = hmm_cell.transitioner.make_A()
            init = hmm_cell.transitioner.make_initial_distribution()
            B = hmm_cell.emitter[0].make_B()
            np.testing.assert_almost_equal(A[0], self.A_ref[i], decimal=5) #allow some decimal errors from softmaxes
            np.testing.assert_almost_equal(B[0], self.B_ref[i], decimal=5) 
            np.testing.assert_almost_equal(init[0,0], self.init_ref[i], decimal=5) 
            imp_log_probs = hmm_cell.transitioner.make_implicit_log_probs()[0][0]
            for part_name in imp_log_probs.keys():
                self.assertTrue(part_name in [part[0] for part in hmm_cell.transitioner.implicit_transition_parts[0]], 
                                part_name + " is in the kernel but not under the expected kernel parts. Wrong spelling?")
            for part_name,l in hmm_cell.transitioner.implicit_transition_parts[0]:
                if part_name in imp_log_probs:
                    kernel_length = tf.size(imp_log_probs[part_name]).numpy()
                    self.assertTrue(kernel_length == l, 
                                    "\"" + part_name + "\" implicit probs array has length " + str(kernel_length) + " but kernel length is " + str(l))
        
    def test_multi_models(self):
        models = [0,1]
        hmm_cell, length = self.make_test_cell(models)
        hmm_cell.recurrent_init()
        A = hmm_cell.transitioner.make_A()
        init = hmm_cell.transitioner.make_initial_distribution()
        B = hmm_cell.emitter[0].make_B()
        for i in models:
            q = hmm_cell.num_states[i]
            np.testing.assert_almost_equal(A[i, :q, :q], self.A_ref[i], decimal=5) #allow some decimal errors from softmaxes
            np.testing.assert_almost_equal(B[i, :q], self.B_ref[i], decimal=5) 
            np.testing.assert_almost_equal(init[0,i,:q], self.init_ref[i], decimal=5) 
        
    def test_single_model_forward(self):
        seq = tf.one_hot([[0,1,0,2]], 3)
        for i in range(2):
            hmm_cell, length = self.make_test_cell(i)
            hmm_cell.recurrent_init()
            scaled_forward, loglik = hmm_cell.get_initial_state(batch_size=1)
            init = True
            for j in range(4):
                col = seq[:,j]
                emission_probs = hmm_cell.emission_probs(col)
                log_forward, (scaled_forward, loglik) = hmm_cell(emission_probs, (scaled_forward, loglik), init=init)
                init = False
                ref_forward = self.ref_alpha[i][j]
                ref_scaled_forward = self.ref_scaled_alpha[i][j]
                np.testing.assert_almost_equal(np.exp(log_forward[...,:-1] + log_forward[...,-1:])[0], ref_forward, decimal=4)
                np.testing.assert_almost_equal(scaled_forward[0], ref_scaled_forward, decimal=4)
            np.testing.assert_almost_equal(np.exp(loglik), self.ref_lik[i], decimal=4)
                
    def test_multi_model_forward(self):
        models = [0,1]
        hmm_cell, length = self.make_test_cell(models)
        scaled_forward, loglik = hmm_cell.get_initial_state(batch_size=1)
        seq = tf.one_hot([[0,1,0,2]], 3)
        init = True
        for j in range(4):
            col = np.repeat(seq[:,j], len(models), axis=0)
            emission_probs = hmm_cell.emission_probs(col)
            log_forward, (scaled_forward, loglik) = hmm_cell(emission_probs, (scaled_forward, loglik), init=init)
            init = False
            for i in range(2):
                q = hmm_cell.num_states[i]
                ref_forward = self.ref_alpha[i][j]
                ref_scaled_forward = self.ref_scaled_alpha[i][j]
                np.testing.assert_almost_equal(np.exp(log_forward[...,:-1] + log_forward[...,-1:])[i,:q], ref_forward, decimal=4)
                np.testing.assert_almost_equal(scaled_forward[i,:q], ref_scaled_forward, decimal=4)
        for i in range(2):
            np.testing.assert_almost_equal(np.exp(loglik[i]), self.ref_lik[i], decimal=4)
            
    def test_multi_model_layer(self):
        models = [0,1]
        hmm_cell, length = self.make_test_cell(models)
        hmm_layer = MsaHmmLayer.MsaHmmLayer(hmm_cell, use_prior=False)
        seq = tf.one_hot([[0,1,0,2]], 3)
        #we have to expand the seq dimension
        #we have 2 identical inputs for 2 models respectively
        #the batch size is still 1
        seq = np.repeat(seq[np.newaxis], len(models), axis=0)
        loglik = hmm_layer(seq)[0]
        log_forward,_ = hmm_layer.forward_recursion(seq)
        self.assertEqual(hmm_layer.cell.step_counter.numpy(), 4)
        log_backward = hmm_layer.backward_recursion(seq)
        state_posterior_log_probs = hmm_layer.state_posterior_log_probs(seq)
        for i in range(2):
            q = hmm_cell.num_states[i]
            np.testing.assert_almost_equal(np.exp(loglik[i]), self.ref_lik[i])
            np.testing.assert_almost_equal(np.exp(log_forward)[i,0,:,:q], self.ref_alpha[i], decimal=6)
            np.testing.assert_almost_equal(np.exp(log_backward)[i,0,:,:q], self.ref_beta[i], decimal=6)
            np.testing.assert_almost_equal(np.exp(state_posterior_log_probs[i,0,:,:q]), self.ref_posterior_probs[i], decimal=6)
            
    #also test the hmm layer in a compiled model and use model.predict
    #the cell call is traced once in this case which can cause trouble with the initial step
    def test_multi_model_tf_model(self):
        models = [0,1]
        hmm_cell, length = self.make_test_cell(models)
        hmm_layer = MsaHmmLayer.MsaHmmLayer(hmm_cell, use_prior=False)
        sequences = tf.keras.Input(shape=(None,None,3), name="sequences", dtype=tf.float32)
        loglik = hmm_layer(sequences)[0]
        hmm_tf_model = tf.keras.Model(inputs=[sequences], outputs=[loglik])
        hmm_tf_model.compile()
        seq = tf.one_hot([[0,1,0,2]], 3)
        #we have to expand the seq dimension
        #we have 2 identical inputs for 2 models respectively
        #the batch size is still 1
        seq = np.repeat(seq[np.newaxis], len(models), axis=0)
        loglik = hmm_tf_model.predict(seq)
        for i in range(2):
            np.testing.assert_almost_equal(np.exp(loglik[i]), self.ref_lik[i])
            
    def test_duplication(self):
        models = [0,1]
        hmm_cell, length = self.make_test_cell(models)
        test_shape = [None, None, 3]
        hmm_cell.build(test_shape)
        
        def test_copied_cell(hmm_cell_copy, model_indices):
            emitter_copy = hmm_cell_copy.emitter
            transitioner_copy = hmm_cell_copy.transitioner
            for i,j in enumerate(model_indices):
                #match emissions
                ref_kernel = hmm_cell.emitter[0].emission_kernel[j].numpy()
                kernel_copy = emitter_copy[0].emission_init[i](ref_kernel.shape)
                np.testing.assert_almost_equal(kernel_copy, ref_kernel)
                #insertions
                ref_ins_kernel = hmm_cell.emitter[0].insertion_kernel[j].numpy()
                ins_kernel_copy = emitter_copy[0].insertion_init[i](ref_ins_kernel.shape)
                np.testing.assert_almost_equal(ins_kernel_copy, ref_ins_kernel)
                #transitioners
                for key, ref_kernel in hmm_cell.transitioner.transition_kernel[j].items():
                    ref_kernel = ref_kernel.numpy()
                    kernel_copy = transitioner_copy.transition_init[i][key](ref_kernel.shape)
                    np.testing.assert_almost_equal(kernel_copy, ref_kernel)
                    
        #clone both models
        test_copied_cell(hmm_cell.duplicate(), [0,1])
        
        #clone single model
        for i in range(2):
            test_copied_cell(hmm_cell.duplicate([i]), [i])
            
    def test_parallel_forward(self):
        models = [0,1]
        n = len(models)
        hmm_cell, length = self.make_test_cell(models)
        hmm_layer = MsaHmmLayer.MsaHmmLayer(hmm_cell, use_prior=False, parallel_factor=2)
        seq = tf.one_hot([[0,1,0,2]], 3)
        seq = np.stack([seq]*n)
        hmm_layer.build(seq.shape)
        log_forward,loglik = hmm_layer.forward_recursion(seq)
        self.assertEqual(hmm_layer.cell.step_counter.numpy(), 2)
        for i in range(n):
            q = hmm_cell.num_states[i]
            np.testing.assert_almost_equal(np.exp(loglik[i]), self.ref_lik[i])
            np.testing.assert_almost_equal(np.exp(log_forward)[i,0,:,:q], self.ref_alpha[i], decimal=6)
            
    def test_parallel_backward(self):
        models = [0,1]
        n = len(models)
        hmm_cell, length = self.make_test_cell(models)
        hmm_layer = MsaHmmLayer.MsaHmmLayer(hmm_cell, use_prior=False, parallel_factor=2)
        seq = tf.one_hot([[0,1,0,2]], 3)
        seq = np.stack([seq]*n)
        hmm_layer.build(seq.shape)
        log_backward = hmm_layer.backward_recursion(seq)
        for i in range(n):
            q = hmm_cell.num_states[i]
            np.testing.assert_almost_equal(np.exp(log_backward)[i,0,:,:q], self.ref_beta[i], decimal=6)
        
    def test_parallel_posterior(self):
        models = [0,1]
        n = len(models)
        hmm_cell, length = self.make_test_cell(models)
        hmm_layer = MsaHmmLayer.MsaHmmLayer(hmm_cell, use_prior=False, parallel_factor=2)
        seq = tf.one_hot([[0,1,0,2]], 3)
        seq = np.stack([seq]*n)
        hmm_layer.build(seq.shape)
        state_posterior_log_probs = hmm_layer.state_posterior_log_probs(seq)
        self.assertEqual(hmm_layer.cell.step_counter.numpy(), 2)
        for i in range(2):
            q = hmm_cell.num_states[i]
            np.testing.assert_almost_equal(np.exp(state_posterior_log_probs[i,0,:,:q]), self.ref_posterior_probs[i], decimal=6)

    def test_parallel_longer_seq_batch(self):
        models = [0,1]
        n = len(models)
        hmm_cell, length = self.make_test_cell(models)
        hmm_layer = MsaHmmLayer.MsaHmmLayer(hmm_cell, use_prior=False, parallel_factor=1)
        hmm_layer_parallel = MsaHmmLayer.MsaHmmLayer(hmm_cell, use_prior=False, parallel_factor=4) 
        #set sequence length to 16 so that we have 4 chunks of size 4
        #try one sequence with padding and one without
        seq = tf.one_hot([[0,1,0,1,1,0,1,0,0,1,0,1,1,0,1,0], [0,1,1,1,0,0,1,1,1,2,2,2,2,2,2,2]], 3)
        seq = np.stack([seq]*n)
        hmm_layer.build(seq.shape)
        hmm_layer_parallel.build(seq.shape)
        #non parallel
        log_forward,loglik = hmm_layer.forward_recursion(seq)
        log_backward = hmm_layer.backward_recursion(seq)
        state_posterior_log_probs = hmm_layer.state_posterior_log_probs(seq)
        #parallel 
        log_forward_parallel,loglik_parallel = hmm_layer_parallel.forward_recursion(seq)
        log_backward_parallel = hmm_layer_parallel.backward_recursion(seq)
        state_posterior_log_probs_parallel = hmm_layer_parallel.state_posterior_log_probs(seq)
        self.assertEqual(hmm_layer.cell.step_counter.numpy(), 4)
        for i in range(2):
            q = hmm_cell.num_states[i]
            np.testing.assert_almost_equal(np.exp(loglik[i]), np.exp(loglik_parallel[i]), decimal=6)
            np.testing.assert_almost_equal(np.exp(log_forward)[i,0,:,:q], np.exp(log_forward_parallel)[i,0,:,:q], decimal=6)
            np.testing.assert_almost_equal(np.exp(log_backward)[i,0,:,:q], np.exp(log_backward_parallel)[i,0,:,:q], decimal=6)
            np.testing.assert_almost_equal(np.exp(state_posterior_log_probs[i,0,:,:q]), np.exp(state_posterior_log_probs_parallel[i,0,:,:q]), decimal=5)
            
    def test_parallel_posterior_casino(self):
        y1 = TestSupervisedTraining.get_prediction(1).numpy()
        y2 = TestSupervisedTraining.get_prediction(1).numpy()
        y3 = TestSupervisedTraining.get_prediction(10).numpy()
        np.testing.assert_almost_equal(y1, y2) #if this fails, check if the batches are non-random
        np.testing.assert_almost_equal(y2, y3, decimal=4) #if this fails, parallel != non-parallel

    def test_parallel_viterbi(self):
        models = [0,1]
        n = len(models)
        hmm_cell, length = self.make_test_cell(models)
        seq = tf.one_hot([[0,1,0,2], [1,0,0,0], [1,1,1,1]], 3)
        seq = np.stack([seq]*n)
        viterbi_path_1, gamma_1 = Viterbi.viterbi(seq, hmm_cell, parallel_factor=1, return_variables=True)
        viterbi_path_2, gamma_2 = Viterbi.viterbi(seq, hmm_cell, parallel_factor=2, return_variables=True)
        for i in range(2):
            q = hmm_cell.num_states[i]
            np.testing.assert_almost_equal(np.exp(gamma_1[i,0,:,:q]), self.ref_gamma[i])
            np.testing.assert_almost_equal(np.exp(gamma_2[i,0,:,0,:q]), self.ref_gamma[i][0::2])
            np.testing.assert_almost_equal(np.exp(gamma_2[i,0,:,1,:q]), self.ref_gamma[i][1::2])
            np.testing.assert_almost_equal(viterbi_path_1[i,0], self.ref_viterbi_path[i])
            np.testing.assert_almost_equal(viterbi_path_2[i,0], self.ref_viterbi_path[i])

    def test_parallel_viterbi_long(self):
        models = [0,1]
        n = len(models)
        hmm_cell, length = self.make_test_cell(models)
        np.random.seed(57235782)
        seq = np.random.randint(2, size=(3, 10000))
        seq = tf.one_hot(seq, 3)
        seq = np.stack([seq]*n)
        viterbi_path_1 = Viterbi.viterbi(seq, hmm_cell, parallel_factor=1)
        viterbi_path_100 = Viterbi.viterbi(seq, hmm_cell, parallel_factor=100)
        np.testing.assert_equal(viterbi_path_1.numpy(), viterbi_path_100.numpy())


                
def string_to_one_hot(s):
    i = [SequenceDataset.alphabet.index(aa) for aa in s]
    return tf.one_hot(i, len(SequenceDataset.alphabet)-1)


def get_all_seqs(data : SequenceDataset, num_models):
    indices = np.arange(data.num_seq)
    batch_generator = Training.DefaultBatchGenerator()
    config = Configuration.make_default(num_models)
    batch_generator.configure(data, config)
    ds = Training.make_dataset(indices, 
                                batch_generator, 
                                batch_size=data.num_seq,
                                shuffle=False)
    for (seq, _), _ in ds:
        return seq.numpy()

class TestMSAHMM(unittest.TestCase):
    
    def assert_vec(self, x, y):
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(np.all(x == y), str(x) + " not equal to " + str(y))
    
    
    def test_matrices(self):
        length=32
        hmm_cell = MsaHmmCell.MsaHmmCell(length=length)
        hmm_cell.build((None, None, len(SequenceDataset.alphabet)))
        A = hmm_cell.transitioner.make_A()
        A_sum = np.sum(A, -1)
        for a in A_sum:
            np.testing.assert_almost_equal(a, 1.0, decimal=5)
        B = hmm_cell.emitter[0].make_B()
        B_sum = np.sum(B, -1)
        for b in B_sum:
            np.testing.assert_almost_equal(b, 1.0, decimal=5)
            
            
    def test_cell(self):
        length = 4
        emission_init = Initializers.ConstantInitializer(string_to_one_hot("ACGT").numpy() * 10)
        transition_init = Initializers.make_default_transition_init(MM = 2, 
                                                                    MI = 0,
                                                                    MD = 0,
                                                                    II = 0,
                                                                    IM = 0,
                                                                    DM = 0,
                                                                    DD = 0,
                                                                    FC = 0,
                                                                    FE = 3,
                                                                    R = 0,
                                                                    RF = -1, 
                                                                    T = 0, 
                                                                    scale = 0)
        emitter = Emitter.ProfileHMMEmitter(emission_init = emission_init, 
                                                 insertion_init = tf.keras.initializers.Zeros())
        transitioner = Transitioner.ProfileHMMTransitioner(transition_init = transition_init, 
                                                            flank_init = tf.keras.initializers.Zeros())
        hmm_cell = MsaHmmCell.MsaHmmCell(length, emitter=emitter, transitioner=transitioner)
        hmm_cell.build((None, None, len(SequenceDataset.alphabet)))
        hmm_cell.recurrent_init()
        filename = os.path.dirname(__file__)+"/data/simple.fa"
        with SequenceDataset(filename) as data:
            sequences = get_all_seqs(data, 1)
        sequences = tf.one_hot(sequences, len(SequenceDataset.alphabet))
        self.assertEqual(sequences.shape, (2,1,5,len(SequenceDataset.alphabet)))
        forward, loglik = hmm_cell.get_initial_state(batch_size=2)
        self.assertEqual(loglik[0], 0)
        #next match state should always yield highest probability
        sequences = tf.transpose(sequences, [1,0,2,3])
        emission_probs = hmm_cell.emission_probs(sequences)
        for i in range(length):
            _, (forward, loglik) = hmm_cell(emission_probs[:,:,i], (forward, loglik))
            self.assertEqual(np.argmax(forward[0]), i+1)
        last_loglik = loglik
        #check correct end in match state
        _, (forward, loglik) = hmm_cell(emission_probs[:,:,4], (forward, loglik))
        self.assertEqual(np.argmax(forward[0]), 2*length+2)
        
        hmm_cell.recurrent_init()
        filename = os.path.dirname(__file__)+"/data/length_diff.fa"
        with SequenceDataset(filename) as data:
            sequences = get_all_seqs(data, 1)
        sequences = tf.one_hot(sequences, len(SequenceDataset.alphabet))
        self.assertEqual(sequences.shape, (2,1,10,len(SequenceDataset.alphabet)))
        forward, loglik = hmm_cell.get_initial_state(batch_size=2)
        sequences = tf.transpose(sequences, [1,0,2,3])
        emission_probs = hmm_cell.emission_probs(sequences)
        for i in range(length):
            _, (forward, loglik) = hmm_cell(emission_probs[:,:,i], (forward, loglik))
            self.assertEqual(np.argmax(forward[0]), i+1)
            self.assertEqual(np.argmax(forward[1]), i+1)
        _, (forward, loglik) = hmm_cell(emission_probs[:,:,length], (forward, loglik))
        self.assertEqual(np.argmax(forward[0]), 2*length+2)
        self.assertEqual(np.argmax(forward[1]), 2*length)
        for i in range(4):
            old_loglik = loglik
            _, (forward, loglik) = hmm_cell(emission_probs[:,:,length+1+i], (forward, loglik))
            #the first sequence is shorter and padded with end-symbols
            #the first end symbol in each sequence affects the likelihood, but this is the
            #same constant for all sequences in the batch
            #further padding does not affect the likelihood
            self.assertEqual(old_loglik[0], loglik[0])
            #the second sequence has the motif of the first seq. repeated twice
            #check whether the model loops correctly 
            #looping must yield larger probabilities than using the right flank state
            self.assertEqual(np.argmax(forward[1]), i+1)
            
            
    def test_viterbi(self):
        length = [5, 3]
        emission_init = [Initializers.ConstantInitializer(string_to_one_hot("FELIK").numpy()*20),
                         Initializers.ConstantInitializer(string_to_one_hot("AHC").numpy()*20)]
        transition_init = [Initializers.make_default_transition_init(MM = 0, 
                                                                    MI = 0,
                                                                    MD = 0,
                                                                    II = 0,
                                                                    IM = 0,
                                                                    DM = 0,
                                                                    DD = 0,
                                                                    FC = 0,
                                                                    FE = 0,
                                                                    R = 0,
                                                                    RF = -1, 
                                                                    T = 0, 
                                                                    scale = 0)]*2
        emitter = Emitter.ProfileHMMEmitter(emission_init = emission_init, 
                                                 insertion_init = [tf.keras.initializers.Zeros()]*2)
        transitioner = Transitioner.ProfileHMMTransitioner(transition_init = transition_init, 
                                                            flank_init = [tf.keras.initializers.Zeros()]*2)
        hmm_cell = MsaHmmCell.MsaHmmCell(length, emitter=emitter, transitioner=transitioner)
        hmm_cell.build((None, None, len(SequenceDataset.alphabet)))
        hmm_cell.recurrent_init()
        with SequenceDataset(os.path.dirname(__file__)+"/data/felix.fa") as data:
            ref_seqs = np.array([#model 1
                                [[1,2,3,4,5,12,12,12,12,12,12,12,12,12,12],
                                [0,0,0,1,2,3,4,5,12,12,12,12,12,12,12],
                                [1,2,3,4,5,11,11,11,12,12,12,12,12,12,12],
                                [1,2,3,4,5,10,10,10,1,2,3,4,5,11,12],
                                [0,2,3,4,11,12,12,12,12,12,12,12,12,12,12],
                                [1,2,7,7,7,3,4,5,12,12,12,12,12,12,12],
                                [1,6,6,2,3,8,4,9,9,9,5,12,12,12,12],
                                [1,2,3,8,8,8,4,5,11,11,11,12,12,12,12]], 
                                #model 2
                                [[0,0,0,0,0,8,8,8,8,8,8,8,8,8,8],
                                [1,2,3,7,7,7,7,7,8,8,8,8,8,8,8],
                                [0,0,0,0,0,0,1,3,8,8,8,8,8,8,8],
                                [0,0,0,0,0,1,2,3,6,6,6,6,6,1,8],
                                [1,4,4,4,2,8,8,8,8,8,8,8,8,8,8],
                                [0,0,1,2,3,7,7,7,8,8,8,8,8,8,8],
                                [0,1,2,6,6,1,6,1,2,3,7,8,8,8,8],
                                [0,0,0,1,2,3,6,6,1,2,3,8,8,8,8]]])
            sequences = get_all_seqs(data, 2)
            sequences = np.transpose(sequences, [1,0,2])
            state_seqs_max_lik = Viterbi.viterbi(sequences, hmm_cell).numpy()
            # states : [LEFT_FLANK, MATCH x length, INSERT x length-1, UNANNOTATED_SEGMENT, RIGHT_FLANK, END]
            self.assert_vec(state_seqs_max_lik, ref_seqs)
            #this produces a result identical to above, but runs viterbi batch wise 
            #to avoid memory overflow  
            batch_generator = Training.DefaultBatchGenerator(return_only_sequences=True)
            batch_generator.configure(data, Configuration.make_default(2))
            state_seqs_max_lik2 = Viterbi.get_state_seqs_max_lik(data,
                                                                    batch_generator,
                                                                    np.arange(data.num_seq),
                                                                    batch_size=2,
                                                                    model_ids=[0,1],
                                                                    hmm_cell=hmm_cell)
            self.assert_vec(state_seqs_max_lik2, ref_seqs)
            indices = np.array([0,4,5])
            state_seqs_max_lik3 = Viterbi.get_state_seqs_max_lik(data,
                                                                    batch_generator,
                                                                    indices, #try a subset
                                                                    batch_size=2,
                                                                    model_ids=[0,1],
                                                                    hmm_cell=hmm_cell)
            max_len = np.amax(data.seq_lens[indices])+1

            for i,j in enumerate(indices):
                self.assert_vec(state_seqs_max_lik3[:,i], ref_seqs[:,j, :max_len])
                
                
            indices = np.array([[0,3,0,0,1,0,0,0], 
                                [5,0,6,5,0,2,1,3]]) #skip the left flank 
            
            #first domain hit
            ref_consensus = [#model 1
                            np.array([[0,1,2,3,4],
                                    [3,4,5,6,7],
                                    [0,1,2,3,4],
                                    [0,1,2,3,4],
                                    [-1,1,2,3,-1],
                                    [0,1,5,6,7],
                                    [0,3,4,6,10],
                                    [0,1,2,6,7]]),
                            #model 2
                            np.array([[-1,-1,-1],
                                    [0,1,2],
                                    [6,-1,7], 
                                    [5,6,7],
                                    [0,4,-1],
                                    [2,3,4],
                                    [1,2,-1],
                                    [3,4,5]]) ]
            ref_insertion_lens = [ #model1
                                np.array([[0]*4,
                                        [0]*4,
                                        [0]*4,
                                        [0]*4,
                                        [0]*4,
                                        [0,3,0,0],
                                        [2,0,1,3],
                                        [0,0,3,0]]), 
                                    #model2
                                    np.array([[0,0], 
                                        [0,0],
                                        [0,0],
                                        [0,0],
                                        [3,0],
                                        [0,0],
                                        [0,0],
                                        [0,0]]) ]
            ref_insertion_start = [#model1
                                    np.array([[-1]*4,
                                        [-1]*4,
                                        [-1]*4,
                                        [-1]*4,
                                        [-1]*4,
                                        [-1,2,-1,-1],
                                        [1,-1,5,7],
                                        [-1,-1,3,-1]]),
                                #model2
                                    np.array([[-1,-1], 
                                        [-1,-1],
                                        [-1,-1],
                                        [-1,-1],
                                        [1,-1],
                                        [-1,-1],
                                        [-1,-1],
                                        [-1,-1]]) ]
            ref_finished = np.array([#model 1
                                    [True, True, True, False, True, True, True, True], 
                                    #model 2
                                    [True, True, True, False, True, True, False, False]])
            ref_left_flank_lens = np.array([[0, 3, 0, 0, 1, 0, 0, 0], 
                                            [5, 0, 6, 5, 0, 2, 1, 3]])
            ref_segment_lens = np.array([[0,0,0,3,0,0,0,0],  #model 1
                                        [0,0,0,5,0,0,2,2]]) #model 2
            ref_segment_start = np.array([[5,8,5,5,4,8,11,8],  #model 1
                                        [5,3,8,8,5,5,3,6]]) #model 2
            ref_right_flank_lens = np.array([[0,0,3,1,1,0,0,3],  #model 1
                                        [0,5,0,0,0,3,1,0]]) #model 2
            ref_right_flank_start = np.array([[5,8,5,13,4,8,11,8],  #model 1
                                        [5,3,8,14,5,5,10,11]]) #model 2
            
            s = len(SequenceDataset.alphabet)
            A = SequenceDataset.alphabet.index("A")
            H = SequenceDataset.alphabet.index("H")
            C = SequenceDataset.alphabet.index("C")
            a = SequenceDataset.alphabet.index("A")+s
            h = SequenceDataset.alphabet.index("H")+s
            c = SequenceDataset.alphabet.index("C")+s
            F = SequenceDataset.alphabet.index("F")
            E = SequenceDataset.alphabet.index("E")
            L = SequenceDataset.alphabet.index("L")
            I = SequenceDataset.alphabet.index("I")
            X = SequenceDataset.alphabet.index("K")
            f = SequenceDataset.alphabet.index("F")+s
            e = SequenceDataset.alphabet.index("E")+s
            l = SequenceDataset.alphabet.index("L")+s
            i = SequenceDataset.alphabet.index("I")+s
            x = SequenceDataset.alphabet.index("K")+s
            GAP = s-1
            gap = 2*s-1
                
            ref_left_flank_block = [ np.array([[gap]*3, #model 1
                                            [a,h,c],
                                            [gap]*3, 
                                            [gap]*3, 
                                            [gap, gap, a],
                                            [gap]*3, 
                                            [gap]*3, 
                                            [gap]*3]), 
                                    np.array([[gap,f,e,l,i,x], #model 2
                                            [gap]*6,
                                            [f,e,l,i,x, h], 
                                            [gap,f,e,l,i,x],  
                                            [gap]*6,
                                            [gap,gap,gap,gap,f,e], 
                                            [gap]*5+[f], 
                                            [gap,gap,gap,f,e,l]]) ]
            ref_right_flank_block = [ np.array([[gap]*3,  #model 1
                                            [gap]*3,
                                            [h,a,c], 
                                            [a,gap,gap], 
                                            [h, gap, gap],
                                            [gap]*3, 
                                            [gap]*3, 
                                            [a,h,c]]), 
                                    np.array([[gap]*5,  #model 2
                                            [f,e,l,i,x],
                                            [gap]*5,
                                            [gap]*5,
                                            [gap]*5,
                                            [l,i,x,gap,gap], 
                                            [x]+[gap]*4, 
                                            [gap]*5]) ]
            ref_ins_block = [ np.array([[gap]*2, 
                                    [gap]*2, 
                                    [gap]*2, 
                                    [gap]*2, 
                                    [gap]*2, 
                                    [gap]*2,
                                    [a,h], 
                                    [gap]*2]), 
                            np.array([[gap]*3, 
                                    [gap]*3, 
                                    [gap]*3, 
                                    [gap]*3, 
                                    [e,l,i], 
                                    [gap]*3, 
                                    [gap]*3, 
                                    [gap]*3]) ]
            ref_core_blocks = [ #model 1
                                [np.array([[F,gap,gap,E,gap,gap,gap,L,gap,gap,gap,I,gap,gap,gap,X],
                                        [F,gap,gap,E,gap,gap,gap,L,gap,gap,gap,I,gap,gap,gap,X],
                                        [F,gap,gap,E,gap,gap,gap,L,gap,gap,gap,I,gap,gap,gap,X],
                                        [F,gap,gap,E,gap,gap,gap,L,gap,gap,gap,I,gap,gap,gap,X],
                                        [GAP,gap,gap,E,gap,gap,gap,L,gap,gap,gap,I,gap,gap,gap,GAP],
                                        [F,gap,gap,E,a,h,c,L,gap,gap,gap,I,gap,gap,gap,X],
                                        [F,a,h,E,gap,gap,gap,L,a,gap,gap,I,a,h,c,X],
                                        [F,gap,gap,E,gap,gap,gap,L,a,h,c,I,gap,gap,gap,X]]),
                            np.array([[GAP]*5,
                                        [GAP]*5,
                                        [GAP]*5,
                                        [F,E,L,I,X],
                                        [GAP]*5,
                                        [GAP]*5,
                                        [GAP]*5,
                                        [GAP]*5])], 
                                #model 2
                            [np.array([[GAP, gap, gap, gap, GAP, GAP],
                                        [A,gap, gap, gap, H, C],
                                        [A,gap, gap, gap, GAP, C],
                                        [A,gap, gap, gap, H, C],
                                        [A,e,l,i,H,GAP],
                                        [A,gap, gap, gap, H, C],
                                        [A, gap, gap, gap, H, GAP],
                                        [A, gap, gap, gap, H, C]]),
                            np.array([[GAP]*3,
                                        [GAP]*3,
                                        [GAP]*3,
                                        [A,GAP,GAP],
                                        [GAP]*3,
                                        [GAP]*3,
                                        [A,GAP,GAP],
                                        [A,H,C]])] ]
            ref_num_blocks = [2, 3]
            #second domain hit
            ref_consensus_2 = [ #model 1
                                np.array([[-1]*5]*3 + 
                                    [[8,9,10,11,12]] +
                                    [[-1]*5]*4), 
                                #model 2
                                np.array([[-1]*3]*3 + 
                                        [[13,-1,-1]] +
                                        [[-1]*3]*2 +
                                        [[5,-1,-1], 
                                        [8,9,10]]) ]
            ref_insertion_lens_2 = [ np.array([[0]*4]*8), #model 1
                                    np.array([[0]*2]*8)] #model 2
            ref_insertion_start_2 = [ np.array([[-1]*4]*8), #model 1
                                    np.array([[-1]*2]*8) ] #model 2
            ref_finished_2 = np.array([[True, True, True, True, True, True, True, True], 
                                    [True, True, True, True, True, True, False, True]])
            ref_left_flank_lens_2 = np.array([[0, 3, 0, 0, 1, 0, 0, 0],  #model 1
                                            [5, 0, 6, 5, 0, 2, 1, 3]]) #model 2
            
            def assert_decoding_core_results(decoded, ref):
                for i in range(data.num_seq):
                    for d,r in zip(decoded, ref):
                        self.assert_vec(d[i], r[i]) 
            
            for i in range(len(length)):
                #test decoding
                #test first core block isolated
                decoding_core_results = AlignmentModel.decode_core(length[i], state_seqs_max_lik[i], indices[i])
                assert_decoding_core_results(decoding_core_results, (ref_consensus[i], 
                                                                    ref_insertion_lens[i],
                                                                    ref_insertion_start[i],
                                                                    ref_finished[i])) 
                #test left flank insertions isolated
                left_flank_lens, left_flank_start = AlignmentModel.decode_flank(state_seqs_max_lik[i], 
                                                                            flank_state_id = 0, 
                                                                            indices = np.array([0,0,0,0,0,0,0,0]))
                self.assert_vec(left_flank_lens, ref_left_flank_lens[i])
                self.assert_vec(left_flank_start, np.array([0,0,0,0,0,0,0,0]))
                #test whole decoding
                core_blocks, left_flank, right_flank, unannotated_segments = AlignmentModel.decode(length[i], state_seqs_max_lik[i])
                self.assertEqual(len(core_blocks), ref_num_blocks[i])
                assert_decoding_core_results(core_blocks[0], (ref_consensus[i], 
                                                            ref_insertion_lens[i],
                                                            ref_insertion_start[i],
                                                            ref_finished[i])) 
                assert_decoding_core_results(core_blocks[1], (ref_consensus_2[i], 
                                                            ref_insertion_lens_2[i],
                                                            ref_insertion_start_2[i],
                                                            ref_finished_2[i]))
                self.assert_vec(left_flank[0], ref_left_flank_lens[i])
                self.assert_vec(left_flank[1], np.array([0,0,0,0,0,0,0,0]))
                self.assert_vec(unannotated_segments[0][0], ref_segment_lens[i])
                self.assert_vec(unannotated_segments[0][1], ref_segment_start[i])
                self.assert_vec(right_flank[0], ref_right_flank_lens[i])
                self.assert_vec(right_flank[1], ref_right_flank_start[i])
                
                #test conversion of decoded data to an anctual alignment in table form
                left_flank_block = AlignmentModel.get_insertion_block(sequences[i], 
                                                                    left_flank[0], 
                                                                    np.amax(left_flank[0]),
                                                                    left_flank[1],
                                                                    adjust_to_right=True)
                self.assert_vec(left_flank_block, ref_left_flank_block[i])
                right_flank_block = AlignmentModel.get_insertion_block(sequences[i], 
                                                                    right_flank[0], 
                                                                    np.amax(right_flank[0]),
                                                                    right_flank[1])
                self.assert_vec(right_flank_block, ref_right_flank_block[i])
                ins_lens = core_blocks[0][1][:,0] #just check the first insert for simplicity
                ins_start = core_blocks[0][2][:,0]
                ins_block = AlignmentModel.get_insertion_block(sequences[i], 
                                                            ins_lens, 
                                                            np.amax(ins_lens),
                                                            ins_start)
                self.assert_vec(ins_block, ref_ins_block[i])
                for (C,IL,IS,f), ref in zip(core_blocks, ref_core_blocks[i]):
                    alignment_block = AlignmentModel.get_alignment_block(sequences[i], 
                                                                        C,IL,np.amax(IL, axis=0),IS)
                    self.assert_vec(alignment_block, ref)


    def test_parallel_viterbi(self):
        length = [5, 3]
        emission_init = [Initializers.ConstantInitializer(string_to_one_hot("FELIK").numpy()*20),
                         Initializers.ConstantInitializer(string_to_one_hot("AHC").numpy()*20)]
        transition_init = [Initializers.make_default_transition_init(MM = 0, 
                                                                    MI = 0,
                                                                    MD = 0,
                                                                    II = 0,
                                                                    IM = 0,
                                                                    DM = 0,
                                                                    DD = 0,
                                                                    FC = 0,
                                                                    FE = 0,
                                                                    R = 0,
                                                                    RF = -1, 
                                                                    T = 0, 
                                                                    scale = 0)]*2
        emitter = Emitter.ProfileHMMEmitter(emission_init = emission_init, 
                                                 insertion_init = [tf.keras.initializers.Zeros()]*2)
        transitioner = Transitioner.ProfileHMMTransitioner(transition_init = transition_init, 
                                                            flank_init = [tf.keras.initializers.Zeros()]*2)
        hmm_cell = MsaHmmCell.MsaHmmCell(length, emitter=emitter, transitioner=transitioner)
        hmm_cell.build((None, None, len(SequenceDataset.alphabet)))
        hmm_cell.recurrent_init()
        with SequenceDataset(os.path.dirname(__file__)+"/data/felix.fa") as data:
            ref_seqs = np.array([#model 1
                                [[1,2,3,4,5,12,12,12,12,12,12,12,12,12,12],
                                [0,0,0,1,2,3,4,5,12,12,12,12,12,12,12],
                                [1,2,3,4,5,11,11,11,12,12,12,12,12,12,12],
                                [1,2,3,4,5,10,10,10,1,2,3,4,5,11,12],
                                [0,2,3,4,11,12,12,12,12,12,12,12,12,12,12],
                                [1,2,7,7,7,3,4,5,12,12,12,12,12,12,12],
                                [1,6,6,2,3,8,4,9,9,9,5,12,12,12,12],
                                [1,2,3,8,8,8,4,5,11,11,11,12,12,12,12]], 
                                #model 2
                                [[0,0,0,0,0,8,8,8,8,8,8,8,8,8,8],
                                [1,2,3,7,7,7,7,7,8,8,8,8,8,8,8],
                                [0,0,0,0,0,0,1,3,8,8,8,8,8,8,8],
                                [0,0,0,0,0,1,2,3,6,6,6,6,6,1,8],
                                [1,4,4,4,2,8,8,8,8,8,8,8,8,8,8],
                                [0,0,1,2,3,7,7,7,8,8,8,8,8,8,8],
                                [0,1,2,6,6,1,6,1,2,3,7,8,8,8,8],
                                [0,0,0,1,2,3,6,6,1,2,3,8,8,8,8]]])
            sequences = get_all_seqs(data, 2)
            sequences = np.transpose(sequences, [1,0,2])
            state_seqs_max_lik_1, gamma_1 = Viterbi.viterbi(sequences, hmm_cell, parallel_factor=1, return_variables=True)
            #print("A", gamma_1[1,1,::5])
            state_seqs_max_lik_3, gamma_3 = Viterbi.viterbi(sequences, hmm_cell, parallel_factor=3, return_variables=True)
            state_seqs_max_lik_5, gamma_5 = Viterbi.viterbi(sequences, hmm_cell, parallel_factor=5, return_variables=True)
            np.testing.assert_almost_equal(gamma_1[:,:,::5].numpy(), gamma_3.numpy()[...,0,:], decimal=4)
            np.testing.assert_almost_equal(gamma_1[:,:,4::5].numpy(), gamma_3.numpy()[...,1,:], decimal=4)
            np.testing.assert_almost_equal(gamma_1[:,:,::3].numpy(), gamma_5.numpy()[...,0,:], decimal=4)
            np.testing.assert_almost_equal(gamma_1[:,:,2::3].numpy(), gamma_5.numpy()[...,1,:], decimal=4)
            self.assert_vec(state_seqs_max_lik_3.numpy(), ref_seqs)
            self.assert_vec(state_seqs_max_lik_5.numpy(), ref_seqs)
                
                
    def test_aligned_insertions(self):
        sequences = np.array([[1, 2, 3, 4, 5],
                              [6, 7, 8, 9, 10],
                              [11, 12, 13, 14, 15]])
        lens = np.array([5, 4, 3])
        starts = np.array([0, 1, 2])
        custom_columns = np.array([[0, 1, 2, 3, 4, -1],
                                   [0, 1, 4, 5, -1, -1],
                                   [2, 3, 4, -1, -1, -1]])
        block = AlignmentModel.get_insertion_block(sequences, lens, 6, starts, custom_columns=custom_columns)
        expected_block = np.array([[1,  2,  3,  4,  5,  23],
                                   [7,  8,  23, 23, 9,  10 ],
                                   [23, 23, 13, 14, 15, 23]])
        self.assert_vec(block, expected_block+len(SequenceDataset.alphabet))
               
        
    def test_backward(self):
        length = [4]
        transition_kernel_initializers = ref.make_transition_init_A()
        #alphabet: {A,B}
        emission_kernel_initializer = np.log([[0.5, 0.5], [0.1, 0.9], [0.7, 0.3], [0.9, 0.1]])
        emission_kernel_initializer = Initializers.ConstantInitializer(emission_kernel_initializer)
        insertion_kernel_initializer = np.log([0.5, 0.5])
        insertion_kernel_initializer = Initializers.ConstantInitializer(insertion_kernel_initializer)
        emitter = Emitter.ProfileHMMEmitter(emission_init = emission_kernel_initializer, 
                                                 insertion_init = insertion_kernel_initializer)
        transitioner = Transitioner.ProfileHMMTransitioner(transition_init = transition_kernel_initializers)
        hmm_cell = MsaHmmCell.MsaHmmCell(length, dim=2+1, emitter=emitter, transitioner=transitioner)
        seq = tf.one_hot([[[0,1,0]]], 3)
        hmm_layer = MsaHmmLayer.MsaHmmLayer(hmm_cell, 1)
        hmm_layer.build(seq.shape)
        backward_seqs = hmm_layer.backward_recursion(seq)
        backward_ref = np.array([[1.]*11, 
                               [0.49724005, 0.11404998, 0.72149999, 
                                0.73499997, 0.44999999, 0.3       , 
                                0.6       , 0.7       , 0.49931   , 
                                0.30000002, 0.         ]])
        for i in range(2):
            actual = np.exp(backward_seqs[0,0,-(i+1)])
            r = backward_ref[i] + hmm_cell.epsilon
            np.testing.assert_almost_equal(actual, r, decimal=5)
            
            
    def test_posterior_state_probabilities(self):
        train_filename = os.path.dirname(__file__)+"/data/egf.fasta"
        with SequenceDataset(train_filename) as data:
            hmm_cell = MsaHmmCell.MsaHmmCell(32)
            hmm_layer = MsaHmmLayer.MsaHmmLayer(hmm_cell, 1)
            hmm_layer.build((1, None, None, len(SequenceDataset.alphabet)))
            batch_gen = Training.DefaultBatchGenerator()
            batch_gen.configure(data, Configuration.make_default(1))
            indices = tf.range(data.num_seq, dtype=tf.int64)
            ds = Training.make_dataset(indices, batch_gen, batch_size=data.num_seq, shuffle=False)
            for x,_ in ds:
                seq = tf.one_hot(x[0], len(SequenceDataset.alphabet))
                seq = tf.transpose(seq, [1,0,2,3])
                p = hmm_layer.state_posterior_log_probs(seq)
            p = np.exp(p)
            np.testing.assert_almost_equal(np.sum(p, -1), 1., decimal=4)
            
            
    def test_posterior_state_probabilities(self):
        train_filename = os.path.dirname(__file__)+"/data/egf.fasta"
        with SequenceDataset(train_filename) as data:
            hmm_cell = MsaHmmCell.MsaHmmCell(32)
            hmm_layer = MsaHmmLayer.MsaHmmLayer(hmm_cell, 1)
            hmm_layer.build((1, None, None, len(SequenceDataset.alphabet)))
            batch_gen = Training.DefaultBatchGenerator()
            batch_gen.configure(data, Configuration.make_default(1))
            indices = tf.range(data.num_seq, dtype=tf.int64)
            ds = Training.make_dataset(indices, batch_gen, batch_size=data.num_seq, shuffle=False)
            for x,_ in ds:
                seq = tf.one_hot(x[0], len(SequenceDataset.alphabet))
                seq = tf.transpose(seq, [1,0,2,3])
                p = hmm_layer.state_posterior_log_probs(seq)
            p = np.exp(p)
            np.testing.assert_almost_equal(np.sum(p, -1), 1., decimal=4)
        

    def test_sequence_weights(self):
        sequence_weights = np.array([0.1, 0.2, 0.5, 1, 2, 3])
        hmm_layer = MsaHmmLayer.MsaHmmLayer(MsaHmmCell.MsaHmmCell(32), sequence_weights=sequence_weights)
        loglik = np.array([[1,2,3], [4,5,6]])
        indices = np.array([[0,1,2], [3,4,5]])
        weighted_loglik = hmm_layer.apply_sequence_weights(loglik, indices)
        np.testing.assert_equal(np.array([[0.1,0.4,1.5], [4.,10.,18.]]), weighted_loglik)
                
                
                            
class TestAncProbs(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super(TestAncProbs, self).__init__(*args, **kwargs)
        self.paml_all = [Utility.LG_paml] + Utility.LG4X_paml
        self.A = SequenceDataset.alphabet[:20]
    
    def assert_vec(self, x, y, almost=False):
        for i,(a,b) in enumerate(zip(x.shape, y.shape)):
            self.assertTrue(a==b or a==1 or b==1, f"{a} {b} (dim {i})")
        self.assertEqual(x.dtype, y.dtype)
        if almost:
            np.testing.assert_almost_equal(x, y, decimal=5)
        else:
            self.assertTrue(np.all(x == y), str(x) + " not equal to " + str(y))
    
    def parse_a(self, string):
        return np.array([float(x) for x in string.split()], dtype=np.float32)
    
    def assert_equilibrium(self, p):
        np.testing.assert_almost_equal(np.sum(p), 1., decimal=5)
    
    def assert_symmetric(self, matrix):
        self.assertEqual(matrix.shape[-1], matrix.shape[-2])
        n = matrix.shape[-1]
        for i in range(n):
            self.assertEqual(matrix[i,i], 0.)
            for j in range(i+1,n):
                self.assertEqual(matrix[i,j], matrix[j,i])
            
    def assert_rate_matrix(self, Q, p):
        for i in range(Q.shape[0]-1):
            for j in range(Q.shape[0]-1):
                np.testing.assert_almost_equal(Q[i,j] * p[i], 
                                               Q[j,i] * p[j])
        
    def assert_anc_probs(self, anc_prob_seqs, expected_sum, expected_anc_probs=None):
        self.assert_vec( np.sum(anc_prob_seqs, -1, keepdims=True), expected_sum, almost=True)
        if expected_anc_probs is not None:
            self.assert_vec( anc_prob_seqs, expected_anc_probs, almost=True)
        #todo: maybe use known properties of amino acids (e.g. polar, charged, aromatic) to test distributions
        #after some time tau
        
    def assert_anc_probs_layer(self, anc_probs_layer, config):
        anc_probs_layer.build()
        p = anc_probs_layer.make_p()
        R = anc_probs_layer.make_R()
        Q = anc_probs_layer.make_Q()
        self.assertEqual(p.shape[0], config["num_models"])
        self.assertEqual(R.shape[0], config["num_models"])
        self.assertEqual(Q.shape[0], config["num_models"])
        self.assertEqual(p.shape[1], config["num_rate_matrices"])
        self.assertEqual(R.shape[1], config["num_rate_matrices"])
        self.assertEqual(Q.shape[1], config["num_rate_matrices"])
        for model_equi in p:
            for equi in model_equi:
                self.assert_equilibrium(equi)
        for model_exchange in R:
            for exchange in model_exchange:
                self.assert_symmetric(exchange)
        for model_rate,model_equi in zip(Q,p):
            for rate,equi in zip(model_rate,model_equi):
                self.assert_rate_matrix(rate,equi)
                
    def test_paml_parsing(self):
        R1, p1 = Utility.parse_paml(Utility.LG4X_paml[0], self.A)
        true_p1_str = """0.147383 0.017579 0.058208 0.017707 0.026331 
                        0.041582 0.017494 0.027859 0.011849 0.076971 
                        0.147823 0.019535 0.037132 0.029940 0.008059 
                        0.088179 0.089653 0.006477 0.032308 0.097931"""
        true_X_row_1 = "0.295719"
        true_X_row_4 = "1.029289 0.576016 0.251987 0.189008"
        true_X_row_19 = """0.916683 0.102065 0.043986 0.080708 0.885230 
                            0.072549 0.206603 0.306067 0.205944 5.381403 
                            0.561215 0.112593 0.693307 0.400021 0.584622 
                            0.089177 0.755865 0.133790 0.154902"""
        self.assert_vec(p1, self.parse_a(true_p1_str))
        self.assert_vec(R1[1,:1], self.parse_a(true_X_row_1))
        self.assert_vec(R1[4,:4], self.parse_a(true_X_row_4))
        self.assert_vec(R1[19,:19], self.parse_a(true_X_row_19))
        for R,p in map(Utility.parse_paml, self.paml_all, [self.A]*len(self.paml_all)):
            self.assert_equilibrium(p)
            self.assert_symmetric(R)
            
    def test_rate_matrices(self):
        for R,p in map(Utility.parse_paml, self.paml_all, [self.A]*len(self.paml_all)):
            Q = AncProbsLayer.make_rate_matrix(R,p)
            self.assert_rate_matrix(Q, p)
            
    def get_test_configs(self, sequences):
        #assuming sequences only contain the 20 standard AAs
        oh_sequences = tf.one_hot(sequences, 20) 
        anc_probs_init = Initializers.make_default_anc_probs_init(1)
        inv_sp_R = anc_probs_init[1]((1,1,20,20))
        log_p = anc_probs_init[2]((1,1,20))
        p = tf.nn.softmax(log_p)
        cases = []
        for equilibrium_sample in [True, False]:
            for rate_init in [-100., -3., 100.]:
                for num_matrices in [1,3]:
                    case = {}
                    config = Configuration.make_default(1)
                    config["num_models"] = 1
                    config["equilibrium_sample"] = equilibrium_sample
                    config["num_rate_matrices"] = num_matrices
                    if num_matrices > 1:
                        R_stack = np.concatenate([inv_sp_R]*num_matrices, axis=1)
                        p_stack = np.concatenate([log_p]*num_matrices, axis=1)
                        config["encoder_initializer"] = (config["encoder_initializer"][:1] + 
                                                       [Initializers.ConstantInitializer(R_stack),
                                                        Initializers.ConstantInitializer(p_stack)] )
                    config["encoder_initializer"] = ([Initializers.ConstantInitializer(rate_init)] + 
                                                     config["encoder_initializer"][1:])
                    case["config"] = config 
                    if rate_init == -100.:
                        case["expected_anc_probs"] = tf.one_hot(sequences, len(SequenceDataset.alphabet)).numpy()
                    elif rate_init == 100.:
                        anc = np.concatenate([p, np.zeros((1,1,len(SequenceDataset.alphabet)-20), dtype=np.float32)], axis=-1)
                        anc = np.concatenate([anc] * sequences.shape[0] * sequences.shape[1] * sequences.shape[2], axis=1)
                        anc = np.reshape(anc, (sequences.shape[0], sequences.shape[1], sequences.shape[2], len(SequenceDataset.alphabet)))
                        case["expected_anc_probs"] = anc 
                    if equilibrium_sample:
                        expected_freq = tf.linalg.matvec(p, oh_sequences).numpy()
                        case["expected_freq"] = expected_freq
                        if rate_init != -3.:
                            case["expected_anc_probs"] *= expected_freq
                        case["expected_freq"] = np.stack([case["expected_freq"]]*num_matrices, axis=-2)
                    else:
                        case["expected_freq"] = np.ones((), dtype=np.float32)
                    if "expected_anc_probs" in case:
                        case["expected_anc_probs"] = np.stack([case["expected_anc_probs"]]*num_matrices, axis=-2)
                    cases.append(case)
        return cases
    
    def get_simple_seq(self, data):      
        sequences = get_all_seqs(data, 1)[:,:,:-1]
        sequences = np.transpose(sequences, [1,0,2])
        return sequences
            
    def test_anc_probs(self):       
        filename = os.path.dirname(__file__)+"/data/simple.fa"
        with SequenceDataset(filename) as data:          
            sequences = self.get_simple_seq(data)
        n = sequences.shape[1]
        for case in self.get_test_configs(sequences):
            anc_probs_layer = Training.make_anc_probs_layer(n, case["config"])
            self.assert_anc_probs_layer(anc_probs_layer, case["config"])
            anc_prob_seqs = anc_probs_layer(sequences, np.arange(n)[np.newaxis, :]).numpy()
            shape = (case["config"]["num_models"], n, sequences.shape[2], case["config"]["num_rate_matrices"], len(SequenceDataset.alphabet))
            anc_prob_seqs = np.reshape(anc_prob_seqs, shape)
            if "expected_anc_probs" in case:
                self.assert_anc_probs(anc_prob_seqs, case["expected_freq"], case["expected_anc_probs"])
            else:
                self.assert_anc_probs(anc_prob_seqs, case["expected_freq"])
                
        
    def test_encoder_model(self):
        #test if everything still works if adding the encoder-model abstraction layer   
        filename = os.path.dirname(__file__)+"/data/simple.fa"
        with SequenceDataset(filename) as data:       
            sequences = self.get_simple_seq(data)
            n = sequences.shape[1]
            ind = np.arange(n)
            model_length = 10
            batch_gen = Training.DefaultBatchGenerator()
            batch_gen.configure(data, Configuration.make_default(1))
            ds = Training.make_dataset(ind, batch_gen, batch_size=n, shuffle=False)
            for case in self.get_test_configs(sequences):
                # the default emitter initializers expect 25 as last dimension which is not compatible with num_matrix=3
                config = dict(case["config"])
                config["emitter"] = Emitter.ProfileHMMEmitter(emission_init = Initializers.ConstantInitializer(0.), 
                                                                insertion_init = Initializers.ConstantInitializer(0.))
                model = Training.default_model_generator(num_seq=n, 
                                                            effective_num_seq=n, 
                                                            model_lengths=[model_length], 
                                                            config=config,
                                                            data=data)
                am = AlignmentModel(data, 
                                    batch_gen, 
                                    ind, 
                                    batch_size=n, 
                                    model=model)
                self.assert_anc_probs_layer(am.encoder_model.layers[-1], case["config"])
                for x,_ in ds:
                    anc_prob_seqs = am.encoder_model(x).numpy()[:,:,:-1]
                    shape = (case["config"]["num_models"], n, sequences.shape[2], case["config"]["num_rate_matrices"], len(SequenceDataset.alphabet))
                    anc_prob_seqs = np.reshape(anc_prob_seqs, shape)
                if "expected_anc_probs" in case:
                    self.assert_anc_probs(anc_prob_seqs,  case["expected_freq"], case["expected_anc_probs"])
                else:
                    self.assert_anc_probs(anc_prob_seqs,  case["expected_freq"])
                
    def test_transposed(self):
        filename = os.path.dirname(__file__)+"/data/simple.fa"
        with SequenceDataset(filename) as data:     
            sequences = self.get_simple_seq(data)
        n = sequences.shape[1]
        config = Configuration.make_default(1)
        anc_probs_layer = Training.make_anc_probs_layer(1, config)
        msa_hmm_layer = Training.make_msa_hmm_layer(n, 10, config)
        msa_hmm_layer.build((1, None, None, len(SequenceDataset.alphabet)))
        B = msa_hmm_layer.cell.emitter[0].make_B()[0]
        config["transposed"] = True
        anc_probs_layer_transposed = Training.make_anc_probs_layer(n, config)
        anc_prob_seqs = anc_probs_layer_transposed(sequences, np.arange(n)[np.newaxis, :]).numpy()
        shape = (config["num_models"], n, sequences.shape[2], config["num_rate_matrices"], len(SequenceDataset.alphabet))
        anc_prob_seqs = np.reshape(anc_prob_seqs, shape)
        anc_prob_seqs = tf.cast(anc_prob_seqs, B.dtype)
        anc_prob_B = anc_probs_layer(B[tf.newaxis,tf.newaxis,:,:20], rate_indices=[[0]])
        anc_prob_B = tf.squeeze(anc_prob_B)
        prob1 = tf.linalg.matvec(B, anc_prob_seqs)
        oh_seqs = tf.one_hot(sequences, 20, dtype=anc_prob_B.dtype)
        oh_seqs = tf.expand_dims(oh_seqs, -2)
        prob2 = tf.linalg.matvec(anc_prob_B, oh_seqs)
        np.testing.assert_almost_equal(prob1.numpy(), prob2.numpy())
        
            
        
class TestData(unittest.TestCase):
    
    def assert_vec(self, x, y):
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(np.all(x == y), str(x) + " not equal to " + str(y))
        
    def test_default_batch_gen(self):
        filename = os.path.dirname(__file__)+"/data/felix_insert_delete.fa"
        with SequenceDataset(filename) as data:
            batch_gen = Training.DefaultBatchGenerator(shuffle=False)
            batch_gen.configure(data, Configuration.make_default(1))
            test_batches = [[0], [1], [4], [0,2], [0,1,2,3,4], [2,3,4]]
            alphabet = np.array(list(SequenceDataset.alphabet))
            for ind in test_batches:
                ind = np.array(ind)
                ref = [str(data.get_record(i).seq).upper() for i in ind]
                s,i = batch_gen(ind) 
                self.assert_vec(i[:,0], ind)
                for i,(r,j) in enumerate(zip(ref, ind)):
                    self.assertEqual("".join(alphabet[s[i,0,:data.seq_lens[j]]]), r)
        
        
class TestModelSurgery(unittest.TestCase):
    
    
    def assert_vec(self, x, y):
        if isinstance(x, list):
            x = np.array(x)
        if isinstance(y, list):
            y = np.array(y)
        self.assertEqual(x.shape, y.shape, str(x)+" "+str(y))
        self.assertTrue(np.all(x == y), str(x) + " not equal to " + str(y))
        
    def make_test_alignment(self, data : SequenceDataset):
        config = Configuration.make_default(1)
        emission_init = string_to_one_hot("FELIC").numpy()*10
        insert_init= np.squeeze(string_to_one_hot("A") + string_to_one_hot("N"))*10
        transition_init = Initializers.make_default_transition_init(MM = 0, 
                                                                            MI = 0,
                                                                            MD = -1,
                                                                            II = 1,
                                                                            IM = 0,
                                                                            DM = 1,
                                                                            DD = 0,
                                                                            FC = 0,
                                                                            FE = 0,
                                                                            R = 0,
                                                                            RF = 0, 
                                                                            T = 0, 
                                                                           scale = 0)
        transition_init["match_to_match"] = Initializers.ConstantInitializer(0)
        transition_init["match_to_insert"] = Initializers.ConstantInitializer(0)
        transition_init["match_to_delete"] = Initializers.ConstantInitializer(-1)
        transition_init["begin_to_match"] = Initializers.ConstantInitializer([1,0,0,0,0])
        transition_init["match_to_end"] = Initializers.ConstantInitializer(0)
        config["emitter"] = Emitter.ProfileHMMEmitter(emission_init = Initializers.ConstantInitializer(emission_init), 
                                                           insertion_init = Initializers.ConstantInitializer(insert_init))
        config["transitioner"] = Transitioner.ProfileHMMTransitioner(transition_init = transition_init)
        model = Training.default_model_generator(num_seq=10, 
                                                      effective_num_seq=10, 
                                                      model_lengths=[5],
                                                      config=config,
                                                      data=data)
        batch_gen = Training.DefaultBatchGenerator()
        batch_gen.configure(data, config)
        am = AlignmentModel(data, 
                                      batch_gen,
                                      np.arange(data.num_seq),
                                      32, 
                                      model)
        return am
        
    def test_discard_or_expand_positions(self):
        filename = os.path.dirname(__file__)+"/data/felix_insert_delete.fa"
        with SequenceDataset(filename) as data:
            am = self.make_test_alignment(data)
            #a simple alignment to test detection of
            #too sparse columns and too frequent insertions
            ref_seqs = [
                "..........F.-LnnnI-aaaFELnICnnn",             
                "nnnnnnnnnn-.-Lnn.I-aaaF--.ICnnn",
                "..........-.-Lnn.I-...---.--nnn",
                "..........-.--...ICaaaF--.I-nnn",
                "..........FnE-...ICaaaF-LnI-nnn"
            ]
            aligned_sequences = am.to_string(model_index=0, add_block_sep=False)
            for s, ref_s in zip(aligned_sequences, ref_seqs):
                self.assertEqual(s, ref_s)
            self.assertTrue(0 in am.metadata)
            #shape: [number of domain hits, length]
            deletions = np.sum(am.metadata[0].consensus == -1, axis=1)
            self.assert_vec(deletions, [[3,4,2,0,3], [1,4,3,1,3]]) 
            #shape: [number of domain hits, num seq]
            self.assert_vec(am.metadata[0].finished, [[False,False,True,False,False], [True,True,True,True,True]]) 
            #shape: [number of domain hits, num seq, L-1 inner positions]
            self.assert_vec(am.metadata[0].insertion_lens, [[[0, 0, 3, 0],
                                                    [0, 0, 2, 0],
                                                    [0, 0, 2, 0],
                                                    [0, 0, 0, 0],
                                                    [1, 0, 0, 0]],

                                                    [[0, 0, 1, 0],
                                                    [0, 0, 0, 0],
                                                    [0, 0, 0, 0],
                                                    [0, 0, 0, 0],
                                                    [0, 0, 1, 0]]]) 
            pos_expand, expansion_lens, pos_discard = Align.get_discard_or_expand_positions(am)
            pos_expand = pos_expand[0]
            expansion_lens = expansion_lens[0]
            pos_discard = pos_discard[0]
            self.assert_vec(pos_expand, [0,3,5])
            self.assert_vec(expansion_lens, [2,2,3])
            self.assert_vec(pos_discard, [1])
        
        
    def test_extend_mods(self):
        pos_expand = np.array([2,3,5])
        expansion_lens = np.array([9,1,3])
        pos_discard = np.array([4])
        e,l,d = Align.extend_mods(pos_expand, expansion_lens, pos_discard, L=5)
        self.assert_vec(d, [1,2,3])
        self.assert_vec(e, [1,2,4])
        self.assert_vec(l, [10,2,3])
        e,l,d = Align.extend_mods(pos_expand, expansion_lens, pos_discard, L=6, k=1)
        self.assert_vec(d, [2,3,4])
        self.assert_vec(e, [2,3,5])
        self.assert_vec(l, [10,2,3])
        e,l,d = Align.extend_mods(pos_expand, expansion_lens, pos_discard, L=6)
        self.assert_vec(d, [1,2,3,4])
        self.assert_vec(e, [1,2,3,4])
        self.assert_vec(l, [10,2,1,3])
        
        
    def test_update_kernels(self):
        filename = os.path.dirname(__file__)+"/data/felix_insert_delete.fa"
        with SequenceDataset(filename) as data:
            am = self.make_test_alignment(data)
            pos_expand = np.array([2,3,5])
            expansion_lens = np.array([9,1,3])
            pos_discard = np.array([4])
            emission_init2 = [Initializers.ConstantInitializer(string_to_one_hot("A").numpy()*10)]
            transition_init2 = {"begin_to_match" : Initializers.ConstantInitializer(77),
                                "match_to_end" : Initializers.ConstantInitializer(77),
                                "match_to_match" : Initializers.ConstantInitializer(77),
                                "match_to_insert" : Initializers.ConstantInitializer(77),
                                "insert_to_match" : Initializers.ConstantInitializer(77),
                                "insert_to_insert" : Initializers.ConstantInitializer(77),
                                "match_to_delete" : Initializers.ConstantInitializer(77),
                                "delete_to_match" : Initializers.ConstantInitializer(77),
                                "delete_to_delete" : Initializers.ConstantInitializer(77),
                                "left_flank_loop" : Initializers.ConstantInitializer(77),
                                "left_flank_exit" : Initializers.ConstantInitializer(77),
                                "right_flank_loop" : Initializers.ConstantInitializer(77),
                                "right_flank_exit" : Initializers.ConstantInitializer(77),
                                "unannotated_segment_loop" : Initializers.ConstantInitializer(77),
                                "unannotated_segment_exit" : Initializers.ConstantInitializer(77),
                                "end_to_unannotated_segment" : Initializers.ConstantInitializer(77),
                                "end_to_right_flank" : Initializers.ConstantInitializer(77),
                                "end_to_terminal" : Initializers.ConstantInitializer(77) }
            transitions_new, emissions_new,_ = Align.update_kernels(am, 0,
                                                                pos_expand, expansion_lens, pos_discard,
                                                                emission_init2, transition_init2, Initializers.ConstantInitializer(0.0))
        ref_consensus = "FE"+"A"*9+"LAI"+"A"*3
        self.assertEqual(emissions_new[0].shape[0], len(ref_consensus))
        self.assert_vec(emissions_new[0], string_to_one_hot(ref_consensus).numpy()*10)
        self.assert_vec(transitions_new["begin_to_match"], [1,0]+[77]*9+[0,77,0,77,77,77])
        self.assert_vec(transitions_new["match_to_end"], [0,0]+[77]*9+[0,77,0,77,77,77])
        self.assert_vec(transitions_new["match_to_match"], [0]+[77]*15)
        self.assert_vec(transitions_new["match_to_insert"], [0]+[77]*15)
        self.assert_vec(transitions_new["insert_to_match"], [0]+[77]*15)
        self.assert_vec(transitions_new["insert_to_insert"], [1]+[77]*15)
        self.assert_vec(transitions_new["match_to_delete"], [-1,-1]+[77]*15)
        self.assert_vec(transitions_new["delete_to_match"], [1]+[77]*16)
        
        
    def test_apply_mods(self):
        x1 = Align.apply_mods(x=list(range(10)), 
                                       pos_expand=[0,4,7,10], 
                                       expansion_lens=[2,1,2,1], 
                                       pos_discard=[4,6,7,8,9], 
                                       insert_value=55)
        self.assert_vec(x1, [55,55,0,1,2,3,55,5,55,55,55])
        x2 = Align.apply_mods(x=[[1,2,3],[1,2,3]], 
                                       pos_expand=[1], 
                                       expansion_lens=[1], 
                                       pos_discard=[], 
                                       insert_value=[4,5,6])
        self.assert_vec(x2, [[1,2,3],[4,5,6],[1,2,3]])
        
        #remark: This was a special solution that failed under practical circumstances
        #I added the scenario to the tests and reduced it to the problem core below
        L=240
        exp = np.array([0, 1, 10,  25,  26,  27,  30,  31, 36,  66,  71,  89,  95,  
                              102,  123, 124, 125, 126, 138, 154, 183, 192, 193, 203, 204, 
                              221, 222, 223, 231, 232,  233, 234, 240])
        lens = np.array([2, 1, 1, 1, 1, 1, 1, 5, 1, 1, 5, 10, 1, 1, 3, 1, 1, 1, 
                              9, 1, 1, 8, 2, 1, 4, 1, 1, 2, 2, 1, 2, 1, 3])
        dis = np.array([13, 84, 129, 130])
        new_pos_expand, new_expansion_lens, new_pos_discard = Align.extend_mods(exp, lens, dis, L)
        x3 = Align.apply_mods(list(range(L-1)), 
                                       new_pos_expand, new_expansion_lens, new_pos_discard, 
                                       insert_value=-1)
        self.assertEqual(x3.size, L-1-dis.size+np.sum(lens))
        
        #problem core of the above issue, solved by handling as a special case 
        L=5
        exp = np.array([0, 1])
        lens = np.array([1, 1])
        dis = np.array([1])
        new_pos_expand, new_expansion_lens, new_pos_discard = Align.extend_mods(exp, lens, dis, L)
        self.assert_vec(new_pos_expand, [0])
        self.assert_vec(new_expansion_lens, [3])
        self.assert_vec(new_pos_discard, [0,1])
        x3 = Align.apply_mods(list(range(L-1)), 
                                       new_pos_expand, new_expansion_lens, new_pos_discard, 
                                       insert_value=-1)
        self.assert_vec(x3, [-1,-1,-1,2,3])
        
        L=5
        exp = np.array([0, 1])
        lens = np.array([9, 1])
        dis = np.array([])
        new_pos_expand, new_expansion_lens, new_pos_discard = Align.extend_mods(exp, lens, dis, L)
        x3 = Align.apply_mods(list(range(L-1)), 
                                       new_pos_expand, new_expansion_lens, new_pos_discard, 
                                       insert_value=-1)
        self.assertEqual(x3.size, L-1-dis.size+np.sum(lens))
        
        L=10
        exp = np.array([0, L-1])
        lens = np.array([5, 5])
        dis = np.arange(L)
        new_pos_expand, new_expansion_lens, new_pos_discard = Align.extend_mods(exp, lens, dis, L)
        x4 = Align.apply_mods(list(range(L-1)), 
                                       new_pos_expand, new_expansion_lens, new_pos_discard, 
                                       insert_value=-1)
        self.assertEqual(x4.size, L-1-dis.size+np.sum(lens))
        
        
    def test_checked_concat(self):
        e,l,d = Align.extend_mods(pos_expand=np.array([]), 
                                            expansion_lens=np.array([]), 
                                            pos_discard=np.array([0,2,4,5,6,9,10]),
                                            L=11)
        self.assert_vec(e, [1,3])
        self.assert_vec(l, [1,1])
        self.assert_vec(d, [0,1,2,3,4,5,6,8,9])
        e,l,d = Align.extend_mods(pos_expand=np.array([0,4,9,10,11]), 
                                            expansion_lens=np.array([2,1,2,3,1]), 
                                            pos_discard=np.array([]),
                                            L=11)
        self.assert_vec(e, [0,3,8,9,10])
        self.assert_vec(l, [2,2,3,4,1])
        self.assert_vec(d, [3,8,9])
        e,l,d = Align.extend_mods(pos_expand=np.array([]), 
                                            expansion_lens=np.array([]), 
                                            pos_discard=np.array([1]),
                                            L=11, k=1)
        self.assert_vec(e, [1])
        self.assert_vec(l, [1])
        self.assert_vec(d, [1,2])
        e,l,d = Align.extend_mods(pos_expand=np.array([5]), 
                                            expansion_lens=np.array([3]), 
                                            pos_discard=np.array(list(range(8))),
                                            L=8)
        self.assert_vec(e, [4])
        self.assert_vec(l, [2])
        self.assert_vec(d, list(range(7)))
        e,l,d = Align.extend_mods(pos_expand=np.array([5]), 
                                            expansion_lens=np.array([3]), 
                                            pos_discard=np.array(list(range(8))),
                                            L=9, k=1)
        self.assert_vec(e, [5])
        self.assert_vec(l, [3])
        self.assert_vec(d, list(range(8)))
        
        e,l,d = Align.extend_mods(pos_expand=np.array([5]), 
                                            expansion_lens=np.array([3]), 
                                            pos_discard=np.array([0,1,2,4,5,6,7]),
                                            L=8)
        self.assert_vec(e, [4])
        self.assert_vec(l, [3])
        self.assert_vec(d, list(range(7)))
        e,l,d = Align.extend_mods(pos_expand=np.array([5]), 
                                            expansion_lens=np.array([3]), 
                                            pos_discard=np.array([0,1,2,4,5,6,7]),
                                            L=9, k=1)
        self.assert_vec(e, [0,5])
        self.assert_vec(l, [1,3])
        self.assert_vec(d, list(range(8)))
        e,l,d = Align.extend_mods(pos_expand=np.array([0,10]), 
                                            expansion_lens=np.array([5,5]), 
                                            pos_discard=np.arange(10),
                                            L=10)
        self.assert_vec(e, [0,9])
        self.assert_vec(l, [4,5])
        self.assert_vec(d, np.arange(9))
        
    def test_whole_surgery(self):
        pass
        
        

class TestAlignment(unittest.TestCase):
    
    def test_subalignment(self):
        filename = os.path.dirname(__file__)+"/data/felix.fa"
        fasta_file = SequenceDataset(filename)
        length=5
        config = Configuration.make_default(1)
        emission_init = string_to_one_hot("FELIK").numpy()*20
        insert_init= np.squeeze(string_to_one_hot("A") + string_to_one_hot("H") + string_to_one_hot("C"))*20
        config["emitter"] = Emitter.ProfileHMMEmitter(emission_init = Initializers.ConstantInitializer(emission_init), 
                                                           insertion_init = Initializers.ConstantInitializer(insert_init))
        config["transitioner"] = Transitioner.ProfileHMMTransitioner(transition_init =(
                            Initializers.make_default_transition_init(MM = 0, 
                                                                            MI = 0,
                                                                            MD = 0,
                                                                            II = 0,
                                                                            IM = 0,
                                                                            DM = 0,
                                                                            DD = 0,
                                                                            FC = 0,
                                                                            FE = 0,
                                                                            R = 0,
                                                                            RF = 0, 
                                                                            T = 0,
                                                                            scale = 0)))
        model = Training.default_model_generator(num_seq=8, 
                                                      effective_num_seq=8,
                                                      model_lengths=[length], 
                                                      config=config,
                                                      data=fasta_file)
        #subalignment
        subset = np.array([0,2,5])
        batch_gen = Training.DefaultBatchGenerator()
        batch_gen.configure(fasta_file, Configuration.make_default(1))
        #create alignment after building model
        sub_am = AlignmentModel(fasta_file, batch_gen, subset, 32, model)
        subalignment_strings = sub_am.to_string(0, add_block_sep=False)
        ref_subalignment = ["FE...LIK...", "FE...LIKhac", "FEahcLIK..."]
        for s,r in zip(subalignment_strings, ref_subalignment):
            self.assertEqual(s,r)
       
    #this test aims to test the high level alignment function by feeding real world data to it
    #and checking if the resulting alignment meets some friendly thresholds 
    def test_alignment_egf(self):
        train_filename = os.path.dirname(__file__)+"/data/egf.fasta"
        ref_filename = os.path.dirname(__file__)+"/data/egf.ref"
        with SequenceDataset(train_filename) as data:
            with AlignedDataset(ref_filename) as ref_msa:
                ref_subset = np.array([data.seq_ids.index(sid) for sid in ref_msa.seq_ids])
            config = Configuration.make_default(1)
            config["max_surgery_runs"] = 2 #do minimal surgery 
            config["epochs"] = [5,1,5]
            am = Align.fit_and_align(data, 
                                    config=config,
                                    subset=ref_subset, 
                                    verbose=False)
            #some friendly thresholds to check if the alignment does make sense at all
            self.assertTrue(np.amin(am.compute_loglik()) > -70)
            self.assertTrue(am.msa_hmm_layer.cell.length[0] > 25)
            am.to_file(os.path.dirname(__file__)+"/data/egf.out.fasta", 0)
            with AlignedDataset(os.path.dirname(__file__)+"/data/egf.out.fasta") as pred_msa:
                sp = pred_msa.SP_score(ref_msa)
                #based on experience, any half decent hyperparameter choice should yield at least this score
                self.assertTrue(sp > 0.7)

    def test_non_homogeneous_mask(self):
        #non_homogeneous_mask_func
        seq_lens = tf.constant([[3,5,4]])
        class HmmCellMock():
            def __init__(self):
                self.num_models = 1
                self.length = [4]
                self.max_num_states = 11
                self.dtype = tf.float32
        mask = non_homogeneous_mask_func(2, seq_lens, HmmCellMock()).numpy()
        expected_zero_pos = [set([(1,8), (2,8), (3,8), (8,3), (8,4)]),
                            set([(1,8), (8,3), (8,4)]),
                            set([(1,8), (2,8), (8,3), (8,4)])]
        for k in range(3):
            for u in range(11):
                for v in range(11):
                    if (u,v) in expected_zero_pos[k]:
                        self.assertEqual(mask[0,k,u,v], 0, f"Expected 0 at {u},{v}")
                    else:
                        self.assertEqual(mask[0,k,u,v], 1, f"Expected 1 at {u},{v}")
        #hitting a sequence end is a special case, always allow transitions out of the last match
        mask = non_homogeneous_mask_func(4, seq_lens, HmmCellMock()).numpy()
        expected_zero_pos = [set([(1,8), (2,8), (3,8)]),
                            set([(1,8), (2,8), (3,8)]),
                            set([(1,8), (2,8), (3,8)])]
        for k in range(3):
            for u in range(11):
                for v in range(11):
                    if (u,v) in expected_zero_pos[k]:
                        self.assertEqual(mask[0,k,u,v], 0, f"Expected 0 at {u},{v}")
                    else:
                        self.assertEqual(mask[0,k,u,v], 1, f"Expected 1 at {u},{v}")


    def test_find_faulty_sequences(self):
        model_length = 4
        C = 2*model_length
        T = 2*model_length+2
        seq_lens = np.array([3,4,4,2,
                            4,4,4,4,
                            2,5,5,5,
                            3])
        state_seqs_max_lik = np.array([[[1,C,2,T,T], [1,C,2,4,T], [1,2,3,4,T], [1,3,T,T,T], 
                                        [1,2,C,3,T], [1,2,C,1,T], [1,2,C,4,T], [1,2,C,4,T], 
                                        [1,C,T,T,T], [1,2,3,C,5], [1,2,3,4,C], [3,C,C,C,1],
                                        [3,C,1,T,T]]])
        faulty_sequences = find_faulty_sequences(state_seqs_max_lik, model_length, seq_lens)
        np.testing.assert_equal(faulty_sequences, [0, 1, 4, 5, 6, 7, 8])
        
        
class ConsoleTest(unittest.TestCase):
        
    def test_error_handling(self):
        import subprocess
        
        single_seq = "test/data/single_sequence.fasta"
        faulty_format = "test/data/faulty_format.fasta"
        empty_seq = "test/data/empty_sequence.fasta"
        unknown_symbol = "test/data/unknown_symbol.fasta"
        
        single_seq_expected_err = f"File {single_seq} contains only a single sequence."
        faulty_format_expected_err = f"Could not parse any sequences from {faulty_format}."
        empty_seq_expected_err = f"{empty_seq} contains empty sequences."
        unknown_symbol_expected_err = f"Found unknown character(s) in sequence ersteSequenz. Allowed alphabet: {SequenceDataset.alphabet}."
        
        test = subprocess.Popen(["python", "learnMSA.py", "--silent", "-o", "test.out", "-i", single_seq], stderr=subprocess.PIPE)
        output = test.communicate()[1].strip().decode('ascii')
        self.assertEqual(single_seq_expected_err, output[-len(single_seq_expected_err):])
        
        test = subprocess.Popen(["python", "learnMSA.py", "--silent", "-o", "test.out", "-i", faulty_format], stderr=subprocess.PIPE)
        output = test.communicate()[1].strip().decode('ascii')
        self.assertEqual(faulty_format_expected_err, output[-len(faulty_format_expected_err):])
        
        test = subprocess.Popen(["python", "learnMSA.py", "--silent", "-o", "test.out", "-i", empty_seq], stderr=subprocess.PIPE)
        output = test.communicate()[1].strip().decode('ascii')
        self.assertEqual(empty_seq_expected_err, output[-len(empty_seq_expected_err):])
        
        test = subprocess.Popen(["python", "learnMSA.py", "--silent", "-o", "test.out", "-i", unknown_symbol], stderr=subprocess.PIPE)
        output = test.communicate()[1].strip().decode('ascii')
        self.assertTrue(unknown_symbol_expected_err in output)
        

class DirichletTest(unittest.TestCase):
        
    def test_dirichlet_log_pdf_single(self):
        epsilon = 1e-16
        alphas = np.array([ [1., 1., 1.], [1., 2, 3], [50., 50., 50.], [100., 1., 10.] ])
        probs = np.array([[.2, .3, .5], [1.-2*epsilon, epsilon, epsilon], [.8, .1, .1], [.3, .3, .4]])
        expected = np.array([[0.693146, 0.693146, 0.693146, 0.693146],
                            [1.5040779, -106.42974, -2.8134103, 1.0577908], 
                            [-5.509186, -3444.141, -70.27524, 3.4245605], 
                            [-127.1859, -293.1855, -4.427696, -89.05315]])
        q = np.array([1.])
        for e, alpha in zip(expected, alphas):
            alpha = np.expand_dims(alpha, 0)
            log_pdf = DirichletMixture.dirichlet_log_pdf(probs, alpha, q)
            np.testing.assert_almost_equal(log_pdf, e, decimal=3)
            alpha_init = Initializers.ConstantInitializer(AncProbsLayer.inverse_softplus(alpha).numpy())
            mix_init = Initializers.ConstantInitializer(np.log(q))
            mean_log_pdf = DirichletMixture.DirichletMixtureLayer(1,3,
                                                            alpha_init=alpha_init,
                                                            mix_init=mix_init)(probs)
            np.testing.assert_almost_equal(mean_log_pdf, np.mean(e), decimal=3)
            
    def test_dirichlet_log_pdf_mix(self):
        epsilon = 1e-16
        alpha = np.array([ [1., 1., 1.], [1., 2, 3], [50., 50., 50.], [100., 1., 10.] ])
        probs = np.array([[.2, .3, .5], [1.-2*epsilon, epsilon, epsilon], [.8, .1, .1], [.3, .3, .4]])
        
        expected = np.array([0.48613059, -0.69314836, -0.65780917,  2.1857463])
        q = np.array([0.25, 0.25, 0.25, 0.25])
        log_pdf = DirichletMixture.dirichlet_log_pdf(probs, alpha, q)
        np.testing.assert_almost_equal(log_pdf, expected, decimal=3)
        alpha_init = Initializers.ConstantInitializer(AncProbsLayer.inverse_softplus(alpha).numpy())
        mix_init = Initializers.ConstantInitializer(np.log(q))
        mean_log_pdf = DirichletMixture.DirichletMixtureLayer(4, 3,
                                                        alpha_init=alpha_init,
                                                        mix_init=mix_init)(probs)
        np.testing.assert_almost_equal(mean_log_pdf, np.mean(expected), decimal=3)
        
        expected2 = np.array([0.39899244, 0.33647106, 0.33903092, 1.36464418])
        q2 = np.array([0.7, 0.02, 0.08, 0.2])
        log_pdf2 = DirichletMixture.dirichlet_log_pdf(probs, alpha, q2)
        np.testing.assert_almost_equal(log_pdf2, expected2, decimal=3)
        mix_init2 = Initializers.ConstantInitializer(np.log(q2))
        mean_log_pdf2 = DirichletMixture.DirichletMixtureLayer(4, 3,
                                                        alpha_init=alpha_init,
                                                        mix_init=mix_init2)(probs)
        np.testing.assert_almost_equal(mean_log_pdf2, np.mean(expected2), decimal=3)
        
        
        
class TestPriors(unittest.TestCase):
        
    def test_amino_acid_match_prior(self):
        prior = Priors.AminoAcidPrior(dtype=tf.float64)
        prior.build([])
        model_lengths = [2,5,3]
        num_models = len(model_lengths)
        max_len = max(model_lengths)
        max_num_states = 2*max_len+3
        B = np.random.rand(3, max_num_states, 26)
        B /= np.sum(B, -1, keepdims=True)
        pdf = prior(B, lengths=model_lengths)
        self.assertEqual(pdf.shape, (num_models, max_len))
        for i,l in enumerate(model_lengths):
            np.testing.assert_equal(pdf[i,l:].numpy(), 0.)
        
        
        
class TestModelToFile(unittest.TestCase):
        
    def test_model_to_file(self):
        test_filepath = "test/data/test_model"
        
        #remove saved models from previous tests
        shutil.rmtree(test_filepath, ignore_errors=True)
        shutil.rmtree(test_filepath+".zip", ignore_errors=True)
        
        #make a model with some custom parameters to save
        model_len = 10
        custom_transition_init = Initializers.make_default_transition_init(MM=7, 
                                                                                     MI=-5, 
                                                                                     MD=2, 
                                                                                     II=5, 
                                                                                     IM=12, 
                                                                                     DM=3, 
                                                                                     DD=22,
                                                                                     FC=6, 
                                                                                     FE=7,
                                                                                     R=8, 
                                                                                     RF=-2, 
                                                                                     T=-10)
        custom_flank_init = Initializers.ConstantInitializer(2)
        em_init_np = np.random.rand(model_len, len(SequenceDataset.alphabet)-1)
        em_init_np[2:6] = string_to_one_hot("ACGT").numpy()*20.
        custom_emission_init = Initializers.ConstantInitializer(em_init_np)
        custom_insertion_init = Initializers.ConstantInitializer(np.random.rand(len(SequenceDataset.alphabet)-1))
        encoder_initializer = Initializers.make_default_anc_probs_init(1)
        encoder_initializer[0] = Initializers.ConstantInitializer(np.random.rand(1, 2))
        config = Configuration.make_default(1)
        config["transitioner"] = Transitioner.ProfileHMMTransitioner(custom_transition_init, custom_flank_init)
        config["emitter"] = Emitter.ProfileHMMEmitter(custom_emission_init, custom_insertion_init)
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
        transition_kernel = {key : kernel.numpy() 
                                for key, kernel in msa_hmm_layer.cell.transitioner.transition_kernel[0].items()}
        flank_init_kernel = msa_hmm_layer.cell.transitioner.flank_init_kernel[0].numpy()
        tau_kernel = anc_probs_layer.tau_kernel.numpy()
        seq = np.random.randint(25, size=(1,1,17))
        seq[:,:,-1] = 25
        loglik = model([seq, np.array([[0]])])[1].numpy()
        
        #make alignment and save
        with SequenceDataset("test/data/simple.fa") as data:
            batch_gen = Training.DefaultBatchGenerator()
            batch_gen.configure(data, config)
            ind = np.array([0,1])
            batch_size = 2
            am = AlignmentModel(data, batch_gen, ind, batch_size, model)
            tf.get_logger().setLevel('ERROR') #prints some info and unrelevant warnings
            am.write_models_to_file(test_filepath)
            tf.get_logger().setLevel('WARNING')
            
            #remember how the decoded MSA looks and delete the alignment object
            #todo: the MSA is currently nonsense, but it should be enough to test in Viterbi runs are consistent
            tf.get_logger().setLevel('ERROR') #prints expected warnings about retracing
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
            self.assertEqual(msa_str, msa_str_from_deserialized_model)


class TestMvnMixture(unittest.TestCase):

    def test_mvn_single_diag_only(self):
        np.random.seed(77)
        mu = np.array([1., 2, 3, 4, 5], dtype=np.float32)
        d = mu.size
        scale_diag = np.array([.1, .5, 1, 2, 3], dtype=np.float32)
        scale = np.diag(scale_diag)
        inputs = np.random.multivariate_normal(mu, scale, size=100).astype(np.float32)
        # compute a reference assuming that tfp uses a correct implementation
        # precomputed to not have tfp as a dependency
        # import tensorflow_probability as tfp
        # ref_dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=scale_diag)
        # ref_log_pdf = ref_dist.log_prob(inputs).numpy()
        ref_log_pdf = np.array([-4.9309063, -10.387703, -7.2909403, -4.436602, -27.024857, -5.253654, -8.531678, -10.450501, 
                                -7.292756, -9.677152, -5.506526, -12.509983, -46.371372, -4.3663635, -5.6152353, -7.4584303, 
                                -10.1104355, -8.395979, -4.0930266, -8.157899, -6.469728, -4.999018, -23.696964, -5.858695, 
                                -9.351412, -20.090115, -10.501094, -7.5617256, -11.118088, -6.100891, -8.797119, -8.742572, 
                                -7.511483, -19.419811, -16.507904, -14.794913, -6.568233, -9.369138, -8.830558, -6.8634834, 
                                -5.3114634, -8.607314, -10.14441, -13.3842535, -25.595154, -9.565649, -7.575345, -11.253255, 
                                -12.537072, -5.991749, -7.7170773, -6.0476046, -7.0488434, -5.262626, -6.590644, -11.621238, 
                                -7.5096426, -4.417821, -6.5862513, -5.6849046, -27.452187, -8.345725, -15.220712, -14.761422, 
                                -10.24355, -26.648506, -4.4250154, -7.253064, -9.48193, -9.329909, -10.536048, -20.788212, 
                                -7.585931, -9.671636, -16.005692, -7.0307503, -6.0479403, -9.651827, -9.062091, -5.6201506, 
                                -6.406321, -5.8290544, -5.5334306, -7.816556, -13.276981, -12.612909, -15.576953, -6.67165, 
                                -7.9874716, -11.401611, -12.275256, -4.94542, -8.66773, -10.191403, -4.8015766, -15.398995, 
                                -6.0817037, -13.620152, -4.935639, -5.3541765])
        # reshape to match the expected shapes
        mu = np.reshape(mu, (1,1,1,d))
        scale_diag = np.reshape(scale_diag, (1,1,1,d))
        inputs = np.expand_dims(inputs, 0)
        # compute the log_prob using the custom implementation
        kernel = Utility.make_kernel(mu, scale_diag)
        dist = MvnMixture.MvnMixture(dim = d, kernel = kernel, diag_only = True)
        log_pdf = dist.log_pdf(inputs)
        # compare the results
        np.testing.assert_almost_equal(log_pdf[0,:,0].numpy(), ref_log_pdf, decimal=5)

    def test_mvn_single_full(self):
        np.random.seed(1000)
        mu = np.array([1., 2, 3, 4, 5], dtype=np.float32)
        d = mu.size
        scale = np.random.rand(d,d).astype(np.float32)
        scale = np.matmul(scale, scale.T)
        #make triangular lower
        inputs = np.random.multivariate_normal(mu, scale, size=100).astype(np.float32)
        scale = np.tril(scale)
        # compute a reference assuming that tfp uses a correct implementation
        # precomputed to not have tfp as a dependency
        # import tensorflow_probability as tfp
        # ref_dist = tfp.distributions.MultivariateNormalTriL(loc=mu, scale_tril=scale)
        # ref_log_pdf = ref_dist.log_prob(inputs).numpy()
        ref_log_pdf = np.array([-7.7539816, -7.4303236, -7.687662, -7.848358, -8.034961, -7.8960657, -7.912344, 
                                -8.077688, -7.4278164, -7.7296476, -7.9770775, -7.7325077, -7.557103, -7.9264936, 
                                -7.5739584, -7.5619946, -7.9560127, -8.073469, -8.438643, -7.351403, -7.7048674, 
                                -7.4203863, -7.5072117, -7.4718685, -7.360958, -7.8579006, -7.5296745, -7.532502, 
                                -7.8499794, -7.718437, -7.3419724, -7.717458, -7.4757233, -7.524334, -7.77816, 
                                -7.513461, -7.831909, -7.32905, -7.4544945, -7.423876, -8.126367, -8.259831, -8.164715, 
                                -7.6520863, -7.459009, -7.765686, -7.4940863, -7.547223, -7.6232376, -7.4277477, 
                                -7.6923866, -7.448251, -7.502944, -7.5673, -7.422387, -7.4737186, -7.7795763, 
                                -7.917593, -7.46449, -7.79692, -7.6767826, -7.386978, -7.58654, -8.014507, -8.283238, 
                                -7.520664, -7.4706373, -7.746441, -7.4981394, -7.2982836, -7.358142, -7.3786783, 
                                -7.4688787, -8.411664, -8.404474, -7.556675, -9.168212, -8.249679, -7.396226, -7.7976904, 
                                -8.388175, -7.284189, -7.3497314, -7.3587933, -7.486101, -7.7001696, -7.9018383, -7.7343836, 
                                -8.025629, -7.430558, -7.668483, -7.534708, -7.740678, -7.649802, -7.9278097, -7.4801702, 
                                -7.7048597, -7.749136, -9.000921, -8.478548])
        # reshape to match the expected shapes
        mu = np.reshape(mu, (1,1,1,d))
        scale = np.reshape(scale, (1,1,1,d,d))
        inputs = np.expand_dims(inputs, 0)
        # compute the log_prob using the custom implementation
        kernel = Utility.make_kernel(mu, scale)
        dist = MvnMixture.MvnMixture(dim = d, kernel = kernel, diag_only = False)
        log_pdf = dist.log_pdf(inputs)
        # compare the results
        np.testing.assert_almost_equal(log_pdf[0,:,0].numpy(), ref_log_pdf, decimal=2)
        

class TestLanguageModelExtension(unittest.TestCase):

    def test_embedding_cache(self):
        seq_lens = np.array([5, 11, 17, 4, 5])
        dim = 32
        def compute_emb_func(indices):
            batch = np.zeros((indices.size, np.amax(seq_lens[indices]), dim), dtype=np.float32)
            for i,j in enumerate(indices):
                batch[i, :seq_lens[j]] = (j+1) * np.ones((seq_lens[j], dim), dtype=np.float32)
            return batch
        cache = EmbeddingCache.EmbeddingCache(seq_lens, dim)
        num_calls = [0, 0]
        def batch_size_callback(L):
            if L > 10:
                num_calls[0] += 1
                return 1
            else:
                num_calls[1] += 1
                return 2
        self.assertFalse(cache.is_filled())
        cache.fill_cache(compute_emb_func, batch_size_callback, verbose=False)
        self.assertTrue(cache.is_filled())
        for i in range(len(seq_lens)):
            emb = cache.get_embedding(i)
            np.testing.assert_almost_equal(emb, compute_emb_func(np.array([i]))[0])
        self.assertEqual(num_calls, [2, 2])
        self.assertEqual(np.sum(cache.cache), np.dot(seq_lens, np.arange(1,len(seq_lens)+1)*dim))

    
    def test_regularizer(self):
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
        self.assertTrue( all(r1[0,:-1] == 75 * 9 + 75 * 4) )
        self.assertTrue( r1[0,-1] == 0 )
        self.assertTrue( all(r1[1,:-1] == 75 * 16 + 75 * 25) )
        self.assertTrue( all(r2[0,:-1] == 75 * 9 + 7 * 75 * 4 / 5) )
        self.assertTrue( r2[0,-1] == 0 )
        self.assertTrue( all(r2[1,:-1] == 75 * 16 + 8 * 75 * 25 / 6) )


class TestEmbeddingPretrainingDatapipeline(unittest.TestCase):

    def test_column_occupancies(self):
        fasta = AlignedDataset("test/data/felix_msa.fa")
        column_occupancies = DataPipeline._get_column_occupancies(fasta)
        np.testing.assert_almost_equal(column_occupancies, [1./3, 2./3, 2./3, 1, 1, 2./3, 2./3, 1.])


class TestPretrainingUtilities(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestPretrainingUtilities, self).__init__(*args, **kwargs)
        self.y_true = np.array([[[1., 0, 0, 0, 0],
                            [0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0]],
                            [[1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 0, 1, 0, 0],
                             [0, 0, 0, 1, 0]]])
        self.y_pred = np.array([[[0.6, 0.4, 0, 0, 0],
                            [0, 0.6, 0.4, 0, 0],
                            [0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0]],
                            [[0.6, 0.4, 0, 0, 0],
                             [0, 0.6, 0.4, 0, 0],
                             [0, 0, 0.6, 0.4, 0],
                             [0, 0, 0, 1, 0]]])

    def test_make_masked_categorical(self):
        y_true_masked, y_pred_masked, norm_masked = TrainingUtil.make_masked_categorical(self.y_true, self.y_pred)
        np.testing.assert_almost_equal(y_true_masked, [[1., 0, 0, 0, 0],
                                                        [0, 1, 0, 0, 0],
                                                        [0, 0, 0, 1, 0],
                                                        [1, 0, 0, 0, 0],
                                                        [0, 1, 0, 0, 0],
                                                        [0, 0, 1, 0, 0],
                                                        [0, 0, 0, 1, 0]])
        np.testing.assert_almost_equal(y_pred_masked, [[0.6, 0.4, 0, 0, 0],
                                                        [0, 0.6, 0.4, 0, 0],
                                                        [0, 0, 0, 1, 0],
                                                        [0.6, 0.4, 0, 0, 0],
                                                        [0, 0.6, 0.4, 0, 0],
                                                        [0, 0, 0.6, 0.4, 0],
                                                        [0, 0, 0, 1, 0]])
        np.testing.assert_almost_equal(norm_masked, [3., 3, 3, 4, 4, 4, 4])

    def test_make_masked_binary(self):
        y_true_masked, y_pred_masked, norm_masked = TrainingUtil.make_masked_binary(self.y_true, self.y_pred)
        np.testing.assert_almost_equal(y_true_masked, [[1.], [0], [0], [0], [1], [0], [0], [0], [1], 
                                                       [1], [0], [0], [0], [0], [1], [0], [0], 
                                                       [0], [0], [1], [0], [0], [0], [0], [1]])
        np.testing.assert_almost_equal(y_pred_masked, [[0.6], [0.4], [0], [0], [0.6], [0], [0], [0], [1],
                                                         [0.6], [0.4], [0], [0], [0], [0.6], [0.4], [0], 
                                                         [0], [0], [0.6], [0.4], [0], [0], [0], [1]])
        np.testing.assert_almost_equal(norm_masked, [9.]*9 + [16.]*16)

    def test_masked_loss_categorical(self):
        loss = TrainingUtil.make_masked_func(tf.keras.losses.categorical_crossentropy, categorical=True, name="cee")
        loss_value = loss(self.y_true, self.y_pred)
        np.testing.assert_almost_equal(loss_value, -(2*np.log(0.6)/3 + 3*np.log(0.6) / 4) / 2)

    def test_masked_acc_categorical(self):
        acc = TrainingUtil.make_masked_func(tf.keras.metrics.categorical_accuracy, categorical=True, name="acc")
        acc_value = acc(self.y_true, self.y_pred)
        np.testing.assert_almost_equal(acc_value, 1.)

    def test_masked_loss_binary(self):
        loss = TrainingUtil.make_masked_func(tf.keras.losses.binary_crossentropy, categorical=False, name="bce")
        loss_value = loss(self.y_true, self.y_pred)
        np.testing.assert_almost_equal(loss_value, -(3*np.log(0.6) / 9 + 6*np.log(0.6)/16)/2)
                   
        
if __name__ == '__main__':
    unittest.main()