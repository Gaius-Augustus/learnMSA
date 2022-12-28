import sys 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import unittest
import numpy as np
import tensorflow as tf
from learnMSA import msa_hmm 
import itertools

class TestFasta(unittest.TestCase):

    def test_parser(self):
        fasta = msa_hmm.fasta.Fasta("test/data/egf.fasta")
        self.assertEqual(fasta.num_seq, 7774)
        self.assertEqual(fasta.aminoacid_seq_str(0), "CDPNPCYNHGTCSLRATGYTCSCLPRYTGEH")
        self.assertEqual(fasta.aminoacid_seq_str(9), "NACDRVRCQNGGTCQLKTLEDYTCSCANGYTGDH")
        self.assertEqual(fasta.aminoacid_seq_str(27), "CNNPCDASPCLNGGTCVPVNAQNYTCTCTNDYSGQN")
        self.assertEqual(fasta.aminoacid_seq_str(-1), "TASCQDMSCSKQGECLETIGNYTCSCYPGFYGPECEYVRE")
        
        fasta2 = msa_hmm.fasta.Fasta("test/data/PF00008_uniprot.fasta")
        self.assertEqual(fasta2.aminoacid_seq_str(0), "PSPCQNGGLCFMSGDDTDYTCACPTGFSG")
        self.assertEqual(fasta2.aminoacid_seq_str(7), "SSPCQNGGMCFMSGDDTDYTCACPTGFSG")
        self.assertEqual(fasta2.aminoacid_seq_str(-1), "CSSSPCNAEGTVRCEDKKGDFLCHCFTGWAGAR")
        
        
def make_test_transition_init():
        inits = [{"begin_to_match" : [0.6, 0.1, 0.1, 0.1],
                  "match_to_end" : [0.01, 0.05, 0.05, 1],
                  "match_to_match" : [0.97, 0.5, 0.6], 
                  "match_to_insert" : [0.01, 0.05, 0.3],
                  "insert_to_match" : [0.5, 0.5, 0.5], 
                  "insert_to_insert" : [0.5, 0.5, 0.5],
                  "match_to_delete" : [0.1, 0.01, 0.4, 0.05], 
                   "delete_to_match" : [0.8, 0.5, 0.8, 1],
                   "delete_to_delete" : [0.2, 0.5, 0.2],
                   #must assume that flaking probs are tied
                   "left_flank_loop" : [0.6], 
                   "left_flank_exit" : [0.4],
                   "right_flank_loop" : [0.6], 
                   "right_flank_exit" : [0.4],
                   "unannotated_segment_loop" : [0.9], 
                   "unannotated_segment_exit" : [0.1],
                   "end_to_unannotated_segment" : [0.2], 
                  "end_to_right_flank" : [0.7], 
                  "end_to_terminal" : [0.1]},
                {"begin_to_match" : [0.7, 0.1, 0.1],
                  "match_to_end" : [0.01, 0.05, 1],
                  "match_to_match" : [0.97, 0.5], 
                  "match_to_insert" : [0.01, 0.05],
                  "insert_to_match" : [0.5, 0.9], 
                  "insert_to_insert" : [0.5, 0.1],
                  "match_to_delete" : [0.1, 0.01, 0.4], 
                   "delete_to_match" : [0.8, 0.5, 1],
                   "delete_to_delete" : [0.2, 0.5],
                   #must assume that flaking probs are tied
                   "left_flank_loop" : [0.6], 
                   "left_flank_exit" : [0.4],
                   "right_flank_loop" : [0.6], 
                   "right_flank_exit" : [0.4],
                   "unannotated_segment_loop" : [0.9], 
                   "unannotated_segment_exit" : [0.1],
                   "end_to_unannotated_segment" : [0.2], 
                  "end_to_right_flank" : [0.7], 
                  "end_to_terminal" : [0.1]}]
        inits = [{part_name : tf.constant_initializer(np.log(p))
                                  for part_name,p in d.items()} for d in inits]
        return inits
    

class TestMsaHmmCell(unittest.TestCase):
   
 
    def test_A(self):
        length = 4
        transition_kernel_initializers = make_test_transition_init()[0]
        emission_kernel_initializer = tf.constant_initializer(np.zeros((length, 2)))
        insertion_kernel_initializer = tf.constant_initializer(np.zeros((2)))
        emitter = msa_hmm.emit.ProfileHMMEmitter(emission_init = emission_kernel_initializer,
                                                insertion_init = insertion_kernel_initializer)
        transitioner = msa_hmm.trans.ProfileHMMTransitioner(transition_init = transition_kernel_initializers)
        hmm_cell = msa_hmm.MsaHmmCell(length, emitter, transitioner)
        hmm_cell.build((None,None,3))
        A = hmm_cell.transitioner.make_A()
        # [LEFT_FLANK, MATCH x length, INSERT x length-1, UNANNOTATED_SEGMENT, RIGHT_FLANK, TERMINAL]
        A_ref = np.zeros((hmm_cell.max_num_states, hmm_cell.max_num_states))
        A_ref[0,0] = .6
        A_ref[0,1] = .4*.6
        A_ref[0,2] = .4*(.1 + .1*.8)
        A_ref[0,3] = .4*(.1 + .1*.2*.5)
        A_ref[0,4] = .4*(.1 + .1*.2*.5*.8)
        A_ref[0,8] = .4*.1*.2*.5*.2*.2
        A_ref[0,9] = .4*.1*.2*.5*.2*.7
        A_ref[0,10] = .4*.1*.2*.5*.2*.1
        A_ref[np.arange(1,4),np.arange(2,5)] = [0.97, 0.5, 0.6]
        A_ref[np.arange(1,4),np.arange(5,8)] = [0.01, 0.05, 0.3]
        A_ref[1,3] = .01*.5
        A_ref[1,4] = .01*.5*.8
        A_ref[1,8] = .2*(.01*.5*.2+.01)
        A_ref[1,9] = .7*(.01*.5*.2+.01)
        A_ref[1,10] = .1*(.01*.5*.2+.01)
        A_ref[2,4] = .4*.8
        A_ref[2,8] = .2*(.4*.2+.05)
        A_ref[2,9] = .7*(.4*.2+.05)
        A_ref[2,10] = .1*(.4*.2+.05)
        A_ref[3,8] = .2*(.05+.05)
        A_ref[3,9] = .7*(.05+.05)
        A_ref[3,10] = .1*(.05+.05)
        A_ref[4,8] = .2
        A_ref[4,9] = .7
        A_ref[4,10] = .1
        A_ref[np.arange(5,8),np.arange(2,5)] = [0.5, 0.5, 0.5]
        A_ref[np.arange(5,8),np.arange(5,8)] = [0.5, 0.5, 0.5]
        A_ref[8,8] = .9 + .1*.1*.2*.5*.2*.2
        A_ref[8,1] = .1*.6
        A_ref[8,2] = .1*(.1+.1*.8)
        A_ref[8,3] = .1*(.1+.1*.2*.5)
        A_ref[8,4] = .1*(.1+.1*.2*.5*.8)
        A_ref[8,9] = .1*.1*.2*.5*.2*.7
        A_ref[8,10] = .1*.1*.2*.5*.2*.1
        A_ref[9,9] = 0.6
        A_ref[9,10] = 0.4
        A_ref[10,10] = 1
        for i in range(hmm_cell.max_num_states):
            for j in range(hmm_cell.max_num_states):
                np.testing.assert_almost_equal(A[0,i,j], 
                                               A_ref[i,j], 
                                               decimal=5,
                                               err_msg=str(i)+","+str(j))
        
        imp_log_probs = hmm_cell.transitioner.make_implicit_log_probs()[0][0]
        for part_name in imp_log_probs.keys():
            self.assertTrue(part_name in [part[0] for part in hmm_cell.transitioner.implicit_transition_parts[0]], 
                            part_name + " is in the kernel but not under the expected kernel parts. Wrong spelling?")
        for part_name,l in hmm_cell.transitioner.implicit_transition_parts[0]:
            if part_name in imp_log_probs:
                kernel_length = tf.size(imp_log_probs[part_name]).numpy()
                self.assertTrue(kernel_length == l, 
                                "\"" + part_name + "\" implicit probs array has length " + str(kernel_length) + " but kernel length is " + str(l))
                
                
    def test_B(self):
        length = 3
        transition_kernel_initializers = make_test_transition_init()[1]
        emission_kernel_initializer = tf.constant_initializer(np.zeros((length, 2)))
        insertion_kernel_initializer = tf.constant_initializer(np.zeros((2)))
        emitter = msa_hmm.emit.ProfileHMMEmitter(emission_init = emission_kernel_initializer,
                                                insertion_init = insertion_kernel_initializer)
        transitioner = msa_hmm.trans.ProfileHMMTransitioner(transition_init = transition_kernel_initializers)
        hmm_cell = msa_hmm.MsaHmmCell(length, emitter, transitioner)
        hmm_cell.build((None,None,3))
        A = hmm_cell.transitioner.make_A()
        # [LEFT_FLANK, MATCH x length, INSERT x length-1, UNANNOTATED_SEGMENT, RIGHT_FLANK, TERMINAL]
        A_ref = np.zeros((hmm_cell.max_num_states, hmm_cell.max_num_states))
        A_ref[0,0] = .6
        A_ref[0,1] = .4*.7
        A_ref[0,2] = .4*(.1 + .1*.8)
        A_ref[0,3] = .4*(.1 + .1*.2*.5)
        A_ref[0,6] = .4*.1*.2*.5*.2
        A_ref[0,7] = .4*.1*.2*.5*.7
        A_ref[0,8] = .4*.1*.2*.5*.1
        A_ref[np.arange(1,3),np.arange(2,4)] = [0.97, 0.5]
        A_ref[np.arange(1,3),np.arange(4,6)] = [0.01, 0.05]
        A_ref[1,3] = .01*.5
        A_ref[1,6] = .2*(.01*.5+.01)
        A_ref[1,7] = .7*(.01*.5+.01)
        A_ref[1,8] = .1*(.01*.5+.01)
        A_ref[2,6] = .2*(.4+.05)
        A_ref[2,7] = .7*(.4+.05)
        A_ref[2,8] = .1*(.4+.05)
        A_ref[3,6] = .2
        A_ref[3,7] = .7
        A_ref[3,8] = .1
        A_ref[np.arange(4,6),np.arange(2,4)] = [0.5, 0.9]
        A_ref[np.arange(4,6),np.arange(4,6)] = [0.5, 0.1]
        A_ref[6,6] = .9 + .1*.1*.2*.5*.2
        A_ref[6,1] = .1*.7
        A_ref[6,2] = .1*(.1+.1*.8)
        A_ref[6,3] = .1*(.1+.1*.2*.5)
        A_ref[6,7] = .1*.1*.2*.5*.7
        A_ref[6,8] = .1*.1*.2*.5*.1
        A_ref[7,7] = 0.6
        A_ref[7,8] = 0.4
        A_ref[8,8] = 1
        for i in range(hmm_cell.max_num_states):
            for j in range(hmm_cell.max_num_states):
                np.testing.assert_almost_equal(A[0,i,j], 
                                               A_ref[i,j], 
                                               decimal=5,
                                               err_msg=str(i)+","+str(j))
        
        imp_log_probs = hmm_cell.transitioner.make_implicit_log_probs()[0][0]
        for part_name in imp_log_probs.keys():
            self.assertTrue(part_name in [part[0] for part in hmm_cell.transitioner.implicit_transition_parts[0]], 
                            part_name + " is in the kernel but not under the expected kernel parts. Wrong spelling?")
        for part_name,l in hmm_cell.transitioner.implicit_transition_parts[0]:
            if part_name in imp_log_probs:
                kernel_length = tf.size(imp_log_probs[part_name]).numpy()
                self.assertTrue(kernel_length == l, 
                                "\"" + part_name + "\" implicit probs array has length " + str(kernel_length) + " but kernel length is " + str(l))
                
                
    def test_multi_model_forward(self):
        length = [4,3]
        transition_kernel_initializers = make_test_transition_init()
        #alphabet: {A,B}
        emission_kernel_initializer1 = np.log([[0.5, 0.5], [0.1, 0.9], [0.7, 0.3], [0.9, 0.1]])
        emission_kernel_initializer2 = np.log([[0.5, 0.5], [0.1, 0.9], [0.7, 0.3]])
        emission_kernel_initializer = [tf.constant_initializer(emission_kernel_initializer1), 
                                       tf.constant_initializer(emission_kernel_initializer2)]
        insertion_kernel_initializer = np.log([0.5, 0.5])
        insertion_kernel_initializer = [tf.constant_initializer(insertion_kernel_initializer)]*2
        emitter = msa_hmm.emit.ProfileHMMEmitter(emission_init = emission_kernel_initializer, 
                                                 insertion_init = insertion_kernel_initializer)
        transitioner = msa_hmm.trans.ProfileHMMTransitioner(transition_init = transition_kernel_initializers,
                                                            flank_init = [msa_hmm.initializers.make_default_flank_init()]*2)
        hmm_cell = msa_hmm.MsaHmmCell(length, emitter, transitioner)
        seq = tf.one_hot([[0,1,0]], 3)
        hmm_cell.build(seq.shape)
        hmm_cell.recurrent_init()
        forward, loglik = hmm_cell.get_initial_state(batch_size=1)
        forward = np.reshape(forward, (2, -1, hmm_cell.max_num_states))
        loglik = np.reshape(loglik, (2, -1, 1))
        ref_forward_scores = np.array([[[0.5, 0.3, 0.09, 
                                         0.055, 0.054, 0, 
                                         0, 0, 0.0002, 
                                         0.0007, 0.0001], 
                                        [0.25, 0.15, 0.009, 
                                         0.0385, 0.0486, 0, 
                                         0, 0, 0.0001, 
                                         0.00035, 0],
                                        [0.1510422, 0.0604229, 0.2963477, 
                                        0.0098184, 0.0075282, 0.0015104, 
                                        0.0004531, 0.01163025, 0.0112615, 
                                        0.0392749, 0],
                                        [0.07689372, 0.0313308, 0.0119539, 
                                         0.184681, 0.173231, 0.001153, 
                                         0.0127664, 0.007433, 0.016715, 
                                         0.0483981, 0]], 
                                      [[0.5, 0.35, 0.09,
                                        0.055, 0, 0,
                                        0.001, 0.0035, 0.0005,
                                        0, 0], 
                                        [0.25   , 0.175  , 0.009  , 
                                         0.0385 , 0.     , 0.     , 
                                         0.0005 , 0.00175, 0.     ,
                                         0, 0],
                                        [0.075     , 0.0350175 , 0.1689831 , 
                                         0.00491415, 0.000875  , 0.000225  , 
                                         0.00484255, 0.01668642, 0.        ,
                                         0, 0],
                                        [0.0225    , 0.01066949, 0.00398916, 
                                         0.06175568, 0.00039384, 0.00423583, 
                                         0.01035781, 0.03363126, 0.        ,
                                         0, 0]]]) 
        for j in range(2):
            np.testing.assert_almost_equal(forward[j:(j+1)], ref_forward_scores[j:(j+1), :1]) 
        for i in range(1,3):
            _, (forward, loglik) = hmm_cell(np.repeat(seq[np.newaxis,:,i-1], len(length), axis=0), 
                                            (forward, loglik))
            forward = np.reshape(forward, (2, -1, hmm_cell.max_num_states))
            loglik = np.reshape(loglik, (2, -1, 1))
            for j in range(2):
                ref = ref_forward_scores[j:(j+1), i:(i+1)]
                np.testing.assert_almost_equal(forward[j:(j+1)], ref / np.sum(ref), decimal=4)
                
    def test_duplication(self):
        length = [4,3]
        transition_kernel_initializers = make_test_transition_init()
        #alphabet: {A,B}
        emission_kernel_initializer1 = np.log([[0.5, 0.5], [0.1, 0.9], [0.7, 0.3], [0.9, 0.1]])
        emission_kernel_initializer2 = np.log([[0.1, 0.9], [0.4, 0.6], [0.5, 0.5]])
        emission_kernel_initializer = [tf.constant_initializer(emission_kernel_initializer1), 
                                       tf.constant_initializer(emission_kernel_initializer2)]
        insertion_kernel_initializer = [tf.constant_initializer(np.log([0.5, 0.5])), 
                                        tf.constant_initializer(np.log([0.3, 0.7]))]
        emitter = msa_hmm.emit.ProfileHMMEmitter(emission_init = emission_kernel_initializer, 
                                                 insertion_init = insertion_kernel_initializer)
        transitioner = msa_hmm.trans.ProfileHMMTransitioner(transition_init = transition_kernel_initializers,
                                                            flank_init = [msa_hmm.initializers.make_default_flank_init()]*2)
        hmm_cell = msa_hmm.MsaHmmCell(length, emitter, transitioner)
        test_shape = [None, None, 3]
        hmm_cell.build(test_shape)
        
        def test_copied_cell(hmm_cell_copy, model_indices):
            emitter_copy = hmm_cell_copy.emitter
            transitioner_copy = hmm_cell_copy.transitioner
            for i,j in enumerate(model_indices):
                #match emissions
                ref_kernel = emitter.emission_kernel[j].numpy()
                kernel_copy = emitter_copy[0].emission_init[i](ref_kernel.shape)
                np.testing.assert_almost_equal(kernel_copy, ref_kernel)
                #insertions
                ref_ins_kernel = emitter.insertion_kernel[j].numpy()
                ins_kernel_copy = emitter_copy[0].insertion_init[i](ref_ins_kernel.shape)
                np.testing.assert_almost_equal(ins_kernel_copy, ref_ins_kernel)
                #transitioners
                for key, ref_kernel in transitioner.transition_kernel[j].items():
                    ref_kernel = ref_kernel.numpy()
                    kernel_copy = transitioner_copy.transition_init[i][key](ref_kernel.shape)
                    np.testing.assert_almost_equal(kernel_copy, ref_kernel)
                    
        #clone both models
        test_copied_cell(hmm_cell.duplicate(), [0,1])
        
        #clone single model
        for i in range(2):
            test_copied_cell(hmm_cell.duplicate([i]), [i])
            
            
        
        
                
                

def string_to_one_hot(s):
    i = [msa_hmm.fasta.alphabet.index(aa) for aa in s]
    return tf.one_hot(i, len(msa_hmm.fasta.alphabet)-1)


def get_all_seqs(fasta_file, num_models):
    indices = np.arange(fasta_file.num_seq)
    batch_generator = msa_hmm.train.DefaultBatchGenerator(fasta_file, num_models)
    ds = msa_hmm.train.make_dataset(indices, 
                                    batch_generator, 
                                    batch_size=fasta_file.num_seq,
                                    shuffle=False)
    for (seq, _), _ in ds:
        return seq.numpy()

class TestMSAHMM(unittest.TestCase):
    
    def assert_vec(self, x, y):
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(np.all(x == y), str(x) + " not equal to " + str(y))
    
    
    def test_matrices(self):
        length=32
        hmm_cell = msa_hmm.MsaHmmCell(length=length)
        hmm_cell.build((None,None,26))
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
        emission_init = tf.constant_initializer(string_to_one_hot("ACGT").numpy() * 10)
        transition_init = msa_hmm.initializers.make_default_transition_init(MM = 2, 
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
        emitter = msa_hmm.emit.ProfileHMMEmitter(emission_init = emission_init, 
                                                 insertion_init = tf.keras.initializers.Zeros())
        transitioner = msa_hmm.trans.ProfileHMMTransitioner(transition_init = transition_init, 
                                                            flank_init = tf.keras.initializers.Zeros())
        hmm_cell = msa_hmm.MsaHmmCell(length, emitter, transitioner)
        hmm_cell.build((None,None, 26))
        hmm_cell.recurrent_init()
        filename = os.path.dirname(__file__)+"/data/simple.fa"
        fasta_file = msa_hmm.fasta.Fasta(filename)
        sequences = get_all_seqs(fasta_file, 1)
        sequences = tf.one_hot(sequences, len(msa_hmm.fasta.alphabet))
        self.assertEqual(sequences.shape, (1,2,5,len(msa_hmm.fasta.alphabet)))
        forward, loglik = hmm_cell.get_initial_state(batch_size=2)
        self.assertEqual(loglik[0], 0)
        #next match state should always yield highest probability
        for i in range(length):
            _, (forward, loglik) = hmm_cell(sequences[:,:,i], (forward, loglik))
            self.assertEqual(np.argmax(forward[0]), i+1)
        last_loglik = loglik
        #check correct end in match state
        _, (forward, loglik) = hmm_cell(sequences[:,:,4], (forward, loglik))
        self.assertEqual(np.argmax(forward[0]), 2*length+2)
        
        hmm_cell.recurrent_init()
        filename = os.path.dirname(__file__)+"/data/length_diff.fa"
        fasta_file = msa_hmm.fasta.Fasta(filename)
        sequences = get_all_seqs(fasta_file,1)
        sequences = tf.one_hot(sequences, len(msa_hmm.fasta.alphabet))
        self.assertEqual(sequences.shape, (1,2,10,len(msa_hmm.fasta.alphabet)))
        forward, loglik = hmm_cell.get_initial_state(batch_size=2)
        for i in range(length):
            _, (forward, loglik) = hmm_cell(sequences[:,:,i], (forward, loglik))
            self.assertEqual(np.argmax(forward[0]), i+1)
            self.assertEqual(np.argmax(forward[1]), i+1)
        _, (forward, loglik) = hmm_cell(sequences[:,:,length], (forward, loglik))
        self.assertEqual(np.argmax(forward[0]), 2*length+2)
        self.assertEqual(np.argmax(forward[1]), 2*length)
        for i in range(4):
            old_loglik = loglik
            _, (forward, loglik) = hmm_cell(sequences[:,:,length+1+i], (forward, loglik))
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
        emission_init = [tf.constant_initializer(string_to_one_hot("FELIX").numpy()*20),
                         tf.constant_initializer(string_to_one_hot("ABC").numpy()*20)]
        transition_init = [msa_hmm.initializers.make_default_transition_init(MM = 0, 
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
        emitter = msa_hmm.emit.ProfileHMMEmitter(emission_init = emission_init, 
                                                 insertion_init = [tf.keras.initializers.Zeros()]*2)
        transitioner = msa_hmm.trans.ProfileHMMTransitioner(transition_init = transition_init, 
                                                            flank_init = [tf.keras.initializers.Zeros()]*2)
        hmm_cell = msa_hmm.MsaHmmCell(length, emitter, transitioner)
        hmm_cell.build((None, None, 26))
        hmm_cell.recurrent_init()
        fasta_file = msa_hmm.fasta.Fasta(os.path.dirname(__file__)+"/data/felix.fa")
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
        sequences = get_all_seqs(fasta_file, 2)
        state_seqs_max_lik = msa_hmm.viterbi.viterbi(sequences, hmm_cell).numpy()
        # states : [LEFT_FLANK, MATCH x length, INSERT x length-1, UNANNOTATED_SEGMENT, RIGHT_FLANK, END]
        self.assert_vec(state_seqs_max_lik, ref_seqs)
        #this produces a result identical to above, but runs viterbi batch wise 
        #to avoid memory overflow  
        batch_generator = msa_hmm.train.DefaultBatchGenerator(fasta_file, 2, return_only_sequences=True)
        state_seqs_max_lik2 = msa_hmm.viterbi.get_state_seqs_max_lik(fasta_file,
                                                                   batch_generator,
                                                                   np.arange(fasta_file.num_seq),
                                                                   batch_size=2,
                                                                   model_ids=[0,1],
                                                                   hmm_cell=hmm_cell)
        self.assert_vec(state_seqs_max_lik2, ref_seqs)
        indices = np.array([0,4,5])
        state_seqs_max_lik3 = msa_hmm.viterbi.get_state_seqs_max_lik(fasta_file,
                                                                   batch_generator,
                                                                   indices, #try a subset
                                                                   batch_size=2,
                                                                   model_ids=[0,1],
                                                                   hmm_cell=hmm_cell)
        max_len = np.amax(fasta_file.seq_lens[indices])+1
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
        
        s = msa_hmm.fasta.s
        A = msa_hmm.fasta.alphabet.index("A")
        B = msa_hmm.fasta.alphabet.index("B")
        C = msa_hmm.fasta.alphabet.index("C")
        a = msa_hmm.fasta.alphabet.index("A")+s
        b = msa_hmm.fasta.alphabet.index("B")+s
        c = msa_hmm.fasta.alphabet.index("C")+s
        F = msa_hmm.fasta.alphabet.index("F")
        E = msa_hmm.fasta.alphabet.index("E")
        L = msa_hmm.fasta.alphabet.index("L")
        I = msa_hmm.fasta.alphabet.index("I")
        X = msa_hmm.fasta.alphabet.index("X")
        f = msa_hmm.fasta.alphabet.index("F")+s
        e = msa_hmm.fasta.alphabet.index("E")+s
        l = msa_hmm.fasta.alphabet.index("L")+s
        i = msa_hmm.fasta.alphabet.index("I")+s
        x = msa_hmm.fasta.alphabet.index("X")+s
        GAP = s-1
        gap = 2*s-1
            
        ref_left_flank_block = [ np.array([[gap]*3, #model 1
                                         [a,b,c],
                                         [gap]*3, 
                                        [gap]*3, 
                                        [gap, gap, a],
                                        [gap]*3, 
                                        [gap]*3, 
                                        [gap]*3]), 
                                np.array([[gap,f,e,l,i,x], #model 2
                                         [gap]*6,
                                         [f,e,l,i,x, b], 
                                        [gap,f,e,l,i,x],  
                                         [gap]*6,
                                        [gap,gap,gap,gap,f,e], 
                                        [gap]*5+[f], 
                                        [gap,gap,gap,f,e,l]]) ]
        ref_right_flank_block = [ np.array([[gap]*3,  #model 1
                                          [gap]*3,
                                          [b,a,c], 
                                          [a,gap,gap], 
                                          [b, gap, gap],
                                          [gap]*3, 
                                          [gap]*3, 
                                          [a,b,c]]), 
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
                                  [a,b], 
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
                                     [F,gap,gap,E,a,b,c,L,gap,gap,gap,I,gap,gap,gap,X],
                                     [F,a,b,E,gap,gap,gap,L,a,gap,gap,I,a,b,c,X],
                                     [F,gap,gap,E,gap,gap,gap,L,a,b,c,I,gap,gap,gap,X]]),
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
                                     [A,gap, gap, gap, B, C],
                                     [A,gap, gap, gap, GAP, C],
                                     [A,gap, gap, gap, B,C],
                                     [A,e,l,i,B,GAP],
                                     [A,gap, gap, gap, B, C],
                                     [A, gap, gap, gap, B, GAP],
                                     [A, gap, gap, gap, B, C]]),
                          np.array([[GAP]*3,
                                    [GAP]*3,
                                    [GAP]*3,
                                    [A,GAP,GAP],
                                    [GAP]*3,
                                    [GAP]*3,
                                    [A,GAP,GAP],
                                    [A,B,C]])] ]
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
            for i in range(fasta_file.num_seq):
                for d,r in zip(decoded, ref):
                    self.assert_vec(d[i], r[i]) 
        
        for i in range(len(length)):
            #test decoding
            #test first core block isolated
            decoding_core_results = msa_hmm.align.decode_core(length[i], state_seqs_max_lik[i], indices[i])
            assert_decoding_core_results(decoding_core_results, (ref_consensus[i], 
                                                                 ref_insertion_lens[i],
                                                                 ref_insertion_start[i],
                                                                 ref_finished[i])) 
            #test left flank insertions isolated
            left_flank_lens, left_flank_start = msa_hmm.align.decode_flank(state_seqs_max_lik[i], 
                                                                          flank_state_id = 0, 
                                                                          indices = np.array([0,0,0,0,0,0,0,0]))
            self.assert_vec(left_flank_lens, ref_left_flank_lens[i])
            self.assert_vec(left_flank_start, np.array([0,0,0,0,0,0,0,0]))
            #test whole decoding
            core_blocks, left_flank, right_flank, unannotated_segments = msa_hmm.align.decode(length[i], state_seqs_max_lik[i])
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
            left_flank_block = msa_hmm.align.get_insertion_block(sequences[i], 
                                                                 left_flank[0], 
                                                                 np.amax(left_flank[0]),
                                                                 left_flank[1],
                                                                 align_to_right=True)
            self.assert_vec(left_flank_block, ref_left_flank_block[i])
            right_flank_block = msa_hmm.align.get_insertion_block(sequences[i], 
                                                                 right_flank[0], 
                                                                 np.amax(right_flank[0]),
                                                                 right_flank[1])
            self.assert_vec(right_flank_block, ref_right_flank_block[i])
            ins_lens = core_blocks[0][1][:,0] #just check the first insert for simplicity
            ins_start = core_blocks[0][2][:,0]
            ins_block = msa_hmm.align.get_insertion_block(sequences[i], 
                                                          ins_lens, 
                                                          np.amax(ins_lens),
                                                          ins_start)
            self.assert_vec(ins_block, ref_ins_block[i])
            for (C,IL,IS,f), ref in zip(core_blocks, ref_core_blocks[i]):
                alignment_block = msa_hmm.align.get_alignment_block(sequences[i], 
                                                                    C,IL,np.amax(IL, axis=0),IS)
                self.assert_vec(alignment_block, ref)
                
               
    def test_backward(self):
        length = [4]
        transition_kernel_initializers = make_test_transition_init()[0]
        #alphabet: {A,B}
        emission_kernel_initializer = np.log([[0.5, 0.5], [0.1, 0.9], [0.7, 0.3], [0.9, 0.1]])
        emission_kernel_initializer = tf.constant_initializer(emission_kernel_initializer)
        insertion_kernel_initializer = np.log([0.5, 0.5])
        insertion_kernel_initializer = tf.constant_initializer(insertion_kernel_initializer)
        emitter = msa_hmm.emit.ProfileHMMEmitter(emission_init = emission_kernel_initializer, 
                                                 insertion_init = insertion_kernel_initializer)
        transitioner = msa_hmm.trans.ProfileHMMTransitioner(transition_init = transition_kernel_initializers)
        hmm_cell = msa_hmm.MsaHmmCell(length, emitter, transitioner)
        seq = tf.one_hot([[[0,1,0]]], 3)
        hmm_layer = msa_hmm.MsaHmmLayer(hmm_cell, 1)
        hmm_layer.build(seq.shape)
        backward_seqs = hmm_layer.backward_recursion(seq)
        backward_ref = np.array([[1.]*11, 
                               [0.49724005, 0.11404998, 0.72149999, 
                                0.73499997, 0.44999999, 0.3       , 
                                0.6       , 0.7       , 0.49931   , 
                                0.30000002, 0.         ]])
        for i in range(2):
            actual = np.exp(backward_seqs[0,0,-(i+1)])
            ref = backward_ref[i] + hmm_cell.epsilon
            np.testing.assert_almost_equal(actual, ref, decimal=5)
            
            
    def test_posterior_state_probabilities(self):
        train_filename = os.path.dirname(__file__)+"/data/egf.fasta"
        fasta_file = msa_hmm.fasta.Fasta(train_filename)
        hmm_cell = msa_hmm.MsaHmmCell(32)
        hmm_layer = msa_hmm.MsaHmmLayer(hmm_cell, 1)
        hmm_layer.build((1, None, None, 26))
        batch_gen = msa_hmm.train.DefaultBatchGenerator(fasta_file, 1)
        indices = tf.range(fasta_file.num_seq, dtype=tf.int64)
        ds = msa_hmm.train.make_dataset(indices, batch_gen, batch_size=fasta_file.num_seq, shuffle=False)
        for x,_ in ds:
            seq = tf.one_hot(x[0], 26)
            p = hmm_layer.state_posterior_log_probs(seq)
        p = np.exp(p)
        np.testing.assert_almost_equal(np.sum(p, -1), 1., decimal=4)
            
                
                
                
                
class TestAncProbs(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super(TestAncProbs, self).__init__(*args, **kwargs)
        self.paml_all = [msa_hmm.anc_probs.LG_paml] + msa_hmm.anc_probs.LG4X_paml
        self.A = msa_hmm.fasta.alphabet[:20]
    
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
        R1, p1 = msa_hmm.anc_probs.parse_paml(msa_hmm.anc_probs.LG4X_paml[0], self.A)
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
        for R,p in map(msa_hmm.anc_probs.parse_paml, self.paml_all, [self.A]*len(self.paml_all)):
            self.assert_equilibrium(p)
            self.assert_symmetric(R)
            
    def test_rate_matrices(self):
        for R,p in map(msa_hmm.anc_probs.parse_paml, self.paml_all, [self.A]*len(self.paml_all)):
            Q = msa_hmm.anc_probs.make_rate_matrix(R,p)
            self.assert_rate_matrix(Q, p)
            
    def get_test_configs(self, sequences):
        #assuming sequences only contain the 20 standard AAs
        oh_sequences = tf.one_hot(sequences, 20) 
        anc_probs_init = msa_hmm.initializers.make_default_anc_probs_init(1)
        inv_sp_R = anc_probs_init[1]((1,1,20,20))
        log_p = anc_probs_init[2]((1,1,20))
        p = tf.nn.softmax(log_p)
        cases = []
        for equilibrium_sample in [True, False]:
            for rate_init in [-100., -3., 100.]:
                for num_matrices in [1,3]:
                    case = {}
                    config = msa_hmm.config.make_default(1)
                    config["num_models"] = 1
                    config["equilibrium_sample"] = equilibrium_sample
                    config["num_rate_matrices"] = num_matrices
                    if num_matrices > 1:
                        R_stack = np.concatenate([inv_sp_R]*num_matrices, axis=1)
                        p_stack = np.concatenate([log_p]*num_matrices, axis=1)
                        config["encoder_initializer"] = (config["encoder_initializer"][:1] + 
                                                       [tf.constant_initializer(R_stack),
                                                        tf.constant_initializer(p_stack)] )
                    config["encoder_initializer"] = ([tf.constant_initializer(rate_init)] + 
                                                     config["encoder_initializer"][1:])
                    case["config"] = config 
                    if rate_init == -100.:
                        case["expected_anc_probs"] = tf.one_hot(sequences, 26).numpy()
                    elif rate_init == 100.:
                        anc = np.concatenate([p, np.zeros((1,1,6), dtype=np.float32)], axis=-1)
                        anc = np.concatenate([anc] * sequences.shape[0] * sequences.shape[1] * sequences.shape[2], axis=1)
                        anc = np.reshape(anc, (sequences.shape[0], sequences.shape[1], sequences.shape[2], 26))
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
    
    def get_simple_seq(self):      
        filename = os.path.dirname(__file__)+"/data/simple.fa"
        fasta_file = msa_hmm.fasta.Fasta(filename)
        sequences = get_all_seqs(fasta_file, 1)[:,:,:-1]
        return sequences, fasta_file
            
    def test_anc_probs(self):                 
        sequences, fasta_file = self.get_simple_seq()
        n = sequences.shape[1]
        for case in self.get_test_configs(sequences):
            anc_probs_layer = msa_hmm.train.make_anc_probs_layer(n, case["config"])
            self.assert_anc_probs_layer(anc_probs_layer, case["config"])
            anc_prob_seqs = anc_probs_layer(sequences, np.arange(n)[np.newaxis, :]).numpy()
            shape = (case["config"]["num_models"], n, sequences.shape[2], case["config"]["num_rate_matrices"], 26)
            anc_prob_seqs = np.reshape(anc_prob_seqs, shape)
            if "expected_anc_probs" in case:
                self.assert_anc_probs(anc_prob_seqs, case["expected_freq"], case["expected_anc_probs"])
            else:
                self.assert_anc_probs(anc_prob_seqs, case["expected_freq"])
                
        
    def test_encoder_model(self):
        #test if everything still works if adding the encoder-model abstraction layer      
        sequences, fasta_file = self.get_simple_seq()
        n = sequences.shape[1]
        ind = np.arange(n)
        model_length = 10
        batch_gen = msa_hmm.train.DefaultBatchGenerator(fasta_file, 1)
        ds = msa_hmm.train.make_dataset(ind, batch_gen, batch_size=n, shuffle=False)
        for case in self.get_test_configs(sequences):
            # the default emitter initializers expect 25 as last dimension which is not compatible with num_matrix=3
            config = dict(case["config"])
            config["emitter"] = msa_hmm.emit.ProfileHMMEmitter(emission_init = tf.constant_initializer(0.), 
                                                               insertion_init = tf.constant_initializer(0.))
            model = msa_hmm.train.default_model_generator(num_seq=n, 
                                                          effective_num_seq=n, 
                                                          model_lengths=[model_length], 
                                                          config=config)
            msa = msa_hmm.Alignment(fasta_file, 
                                    batch_gen, 
                                    ind, 
                                    batch_size=n, 
                                    model=model)
            self.assert_anc_probs_layer(msa.encoder_model.layers[-1], case["config"])
            for x,_ in ds:
                anc_prob_seqs = msa.encoder_model(x).numpy()[:,:,:-1]
                shape = (case["config"]["num_models"], n, sequences.shape[2], case["config"]["num_rate_matrices"], 26)
                anc_prob_seqs = np.reshape(anc_prob_seqs, shape)
            if "expected_anc_probs" in case:
                self.assert_anc_probs(anc_prob_seqs,  case["expected_freq"], case["expected_anc_probs"])
            else:
                self.assert_anc_probs(anc_prob_seqs,  case["expected_freq"])
                
    def test_transposed(self):
        sequences, fasta_file = self.get_simple_seq()
        n = sequences.shape[1]
        config = msa_hmm.config.make_default(1)
        anc_probs_layer = msa_hmm.train.make_anc_probs_layer(1, config)
        msa_hmm_layer = msa_hmm.train.make_msa_hmm_layer(n, 10, config)
        msa_hmm_layer.build((1, None, None, 26))
        B = msa_hmm_layer.cell.emitter[0].make_B()[0]
        config["transposed"] = True
        anc_probs_layer_transposed = msa_hmm.train.make_anc_probs_layer(n, config)
        anc_prob_seqs = anc_probs_layer_transposed(sequences, np.arange(n)[np.newaxis, :]).numpy()
        shape = (config["num_models"], n, sequences.shape[2], config["num_rate_matrices"], 26)
        anc_prob_seqs = np.reshape(anc_prob_seqs, shape)
        anc_prob_seqs = tf.cast(anc_prob_seqs, B.dtype)
        anc_prob_B = anc_probs_layer(B[tf.newaxis,tf.newaxis,:,:20], [[0]])
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
        fasta_file = msa_hmm.fasta.Fasta(filename)
        batch_gen = msa_hmm.train.DefaultBatchGenerator(fasta_file, 1, shuffle=False)
        test_batches = [[0], [1], [4], [0,2], [0,1,2,3,4], [2,3,4]]
        alphabet = np.array(msa_hmm.fasta.alphabet)
        for ind in test_batches:
            ind = np.array(ind)
            ref = [fasta_file.aminoacid_seq_str(i) for i in ind]
            s,i = batch_gen(ind) 
            self.assert_vec(i[0], ind)
            for i,(r,j) in enumerate(zip(ref, ind)):
                self.assertEqual("".join(alphabet[s[0,i,:fasta_file.seq_lens[j]]]), r)
        
        
class TestModelSurgery(unittest.TestCase):
    
    
    def assert_vec(self, x, y):
        if isinstance(x, list):
            x = np.array(x)
        if isinstance(y, list):
            y = np.array(y)
        self.assertEqual(x.shape, y.shape, str(x)+" "+str(y))
        self.assertTrue(np.all(x == y), str(x) + " not equal to " + str(y))
        
    def make_test_alignment(self):
        config = msa_hmm.config.make_default(1)
        emission_init = string_to_one_hot("FELIC").numpy()*10
        insert_init= np.squeeze(string_to_one_hot("A") + string_to_one_hot("N"))*10
        transition_init = msa_hmm.initializers.make_default_transition_init(MM = 0, 
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
        transition_init["match_to_match"] = tf.constant_initializer(0)
        transition_init["match_to_insert"] = tf.constant_initializer(0)
        transition_init["match_to_delete"] = tf.constant_initializer(-1)
        transition_init["begin_to_match"] = tf.constant_initializer([1,0,0,0,0])
        transition_init["match_to_end"] = tf.constant_initializer(0)
        config["emitter"] = msa_hmm.emit.ProfileHMMEmitter(emission_init = tf.constant_initializer(emission_init), 
                                                           insertion_init = tf.constant_initializer(insert_init))
        config["transitioner"] = msa_hmm.trans.ProfileHMMTransitioner(transition_init = transition_init)
        model = msa_hmm.train.default_model_generator(num_seq=10, 
                                                      effective_num_seq=10, 
                                                      model_lengths=[5],
                                                      config=config)
        filename = os.path.dirname(__file__)+"/data/felix_insert_delete.fa"
        fasta_file = msa_hmm.fasta.Fasta(filename)
        batch_gen = msa_hmm.train.DefaultBatchGenerator(fasta_file, 1)
        alignment = msa_hmm.Alignment(fasta_file, 
                                      batch_gen,
                                      np.arange(fasta_file.num_seq),
                                      32, 
                                      model)
        return alignment
        
    def test_discard_or_expand_positions(self):
        alignment = self.make_test_alignment()
        #a simple alignment to test detection of
        #too sparse columns and too frequent insertions
        ref_seqs = [
            "..........F.-LnnnI-aaaFELnICnnn",             
            "nnnnnnnnnn-.-Lnn.I-aaaF--.ICnnn",
            "..........-.-Lnn.I-...---.--nnn",
            "..........-.--...ICaaaF--.I-nnn",
            "..........FnE-...ICaaaF-LnI-nnn"
        ]
        aligned_sequences = alignment.to_string(model_index=0, add_block_sep=False)
        for s, ref_s in zip(aligned_sequences, ref_seqs):
            self.assertEqual(s, ref_s)
        self.assertTrue(0 in alignment.metadata)
        #shape: [number of domain hits, length]
        deletions = np.sum(alignment.metadata[0].consensus == -1, axis=1)
        self.assert_vec(deletions, [[3,4,2,0,3], [1,4,3,1,3]]) 
        #shape: [number of domain hits, num seq]
        self.assert_vec(alignment.metadata[0].finished, [[False,False,True,False,False], [True,True,True,True,True]]) 
        #shape: [number of domain hits, num seq, L-1 inner positions]
        self.assert_vec(alignment.metadata[0].insertion_lens, [[[0, 0, 3, 0],
                                                  [0, 0, 2, 0],
                                                  [0, 0, 2, 0],
                                                  [0, 0, 0, 0],
                                                  [1, 0, 0, 0]],

                                                 [[0, 0, 1, 0],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 1, 0]]]) 
        pos_expand, expansion_lens, pos_discard = msa_hmm.align.get_discard_or_expand_positions(alignment)
        pos_expand = pos_expand[0]
        expansion_lens = expansion_lens[0]
        pos_discard = pos_discard[0]
        self.assert_vec(pos_expand, [2,3,5])
        self.assert_vec(expansion_lens, [2,2,3])
        self.assert_vec(pos_discard, [1])
        
        
    def test_extend_mods(self):
        pos_expand = np.array([2,3,5])
        expansion_lens = np.array([9,1,3])
        pos_discard = np.array([4])
        e,l,d = msa_hmm.align.extend_mods(pos_expand, expansion_lens, pos_discard, L=5)
        self.assert_vec(d, [1,2,3])
        self.assert_vec(e, [1,2,4])
        self.assert_vec(l, [10,2,3])
        e,l,d = msa_hmm.align.extend_mods(pos_expand, expansion_lens, pos_discard, L=6, k=1)
        self.assert_vec(d, [2,3,4])
        self.assert_vec(e, [2,3,5])
        self.assert_vec(l, [10,2,3])
        e,l,d = msa_hmm.align.extend_mods(pos_expand, expansion_lens, pos_discard, L=6)
        self.assert_vec(d, [1,2,3,4])
        self.assert_vec(e, [1,2,3,4])
        self.assert_vec(l, [10,2,1,3])
        
        
    def test_update_kernels(self):
        alignment = self.make_test_alignment()
        pos_expand = np.array([2,3,5])
        expansion_lens = np.array([9,1,3])
        pos_discard = np.array([4])
        emission_init2 = [tf.constant_initializer(string_to_one_hot("A").numpy()*10)]
        transition_init2 = {"begin_to_match" : tf.constant_initializer(77),
                            "match_to_end" : tf.constant_initializer(77),
                            "match_to_match" : tf.constant_initializer(77),
                            "match_to_insert" : tf.constant_initializer(77),
                            "insert_to_match" : tf.constant_initializer(77),
                            "insert_to_insert" : tf.constant_initializer(77),
                            "match_to_delete" : tf.constant_initializer(77),
                            "delete_to_match" : tf.constant_initializer(77),
                            "delete_to_delete" : tf.constant_initializer(77),
                            "left_flank_loop" : tf.constant_initializer(77),
                            "left_flank_exit" : tf.constant_initializer(77),
                            "right_flank_loop" : tf.constant_initializer(77),
                            "right_flank_exit" : tf.constant_initializer(77),
                            "unannotated_segment_loop" : tf.constant_initializer(77),
                            "unannotated_segment_exit" : tf.constant_initializer(77),
                            "end_to_unannotated_segment" : tf.constant_initializer(77),
                            "end_to_right_flank" : tf.constant_initializer(77),
                            "end_to_terminal" : tf.constant_initializer(77) }
        transitions_new, emissions_new,_ = msa_hmm.align.update_kernels(alignment, 0,
                                                              pos_expand, expansion_lens, pos_discard,
                                                              emission_init2, transition_init2, tf.constant_initializer(0.0))
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
        x1 = msa_hmm.align.apply_mods(x=list(range(10)), 
                                       pos_expand=[0,4,7,10], 
                                       expansion_lens=[2,1,2,1], 
                                       pos_discard=[4,6,7,8,9], 
                                       insert_value=55)
        self.assert_vec(x1, [55,55,0,1,2,3,55,5,55,55,55])
        x2 = msa_hmm.align.apply_mods(x=[[1,2,3],[1,2,3]], 
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
        new_pos_expand, new_expansion_lens, new_pos_discard = msa_hmm.align.extend_mods(exp, lens, dis, L)
        x3 = msa_hmm.align.apply_mods(list(range(L-1)), 
                                       new_pos_expand, new_expansion_lens, new_pos_discard, 
                                       insert_value=-1)
        self.assertEqual(x3.size, L-1-dis.size+np.sum(lens))
        
        #problem core of the above issue, solved by handling as a special case 
        L=5
        exp = np.array([0, 1])
        lens = np.array([1, 1])
        dis = np.array([1])
        new_pos_expand, new_expansion_lens, new_pos_discard = msa_hmm.align.extend_mods(exp, lens, dis, L)
        self.assert_vec(new_pos_expand, [0])
        self.assert_vec(new_expansion_lens, [3])
        self.assert_vec(new_pos_discard, [0,1])
        x3 = msa_hmm.align.apply_mods(list(range(L-1)), 
                                       new_pos_expand, new_expansion_lens, new_pos_discard, 
                                       insert_value=-1)
        self.assert_vec(x3, [-1,-1,-1,2,3])
        
        L=5
        exp = np.array([0, 1])
        lens = np.array([9, 1])
        dis = np.array([])
        new_pos_expand, new_expansion_lens, new_pos_discard = msa_hmm.align.extend_mods(exp, lens, dis, L)
        x3 = msa_hmm.align.apply_mods(list(range(L-1)), 
                                       new_pos_expand, new_expansion_lens, new_pos_discard, 
                                       insert_value=-1)
        self.assertEqual(x3.size, L-1-dis.size+np.sum(lens))
        
        L=10
        exp = np.array([0, L-1])
        lens = np.array([5, 5])
        dis = np.arange(L)
        new_pos_expand, new_expansion_lens, new_pos_discard = msa_hmm.align.extend_mods(exp, lens, dis, L)
        x4 = msa_hmm.align.apply_mods(list(range(L-1)), 
                                       new_pos_expand, new_expansion_lens, new_pos_discard, 
                                       insert_value=-1)
        self.assertEqual(x4.size, L-1-dis.size+np.sum(lens))
        
        
    def test_checked_concat(self):
        e,l,d = msa_hmm.align.extend_mods(pos_expand=np.array([]), 
                                            expansion_lens=np.array([]), 
                                            pos_discard=np.array([0,2,4,5,6,9,10]),
                                            L=11)
        self.assert_vec(e, [1,3])
        self.assert_vec(l, [1,1])
        self.assert_vec(d, [0,1,2,3,4,5,6,8,9])
        e,l,d = msa_hmm.align.extend_mods(pos_expand=np.array([0,4,9,10,11]), 
                                            expansion_lens=np.array([2,1,2,3,1]), 
                                            pos_discard=np.array([]),
                                            L=11)
        self.assert_vec(e, [0,3,8,9,10])
        self.assert_vec(l, [2,2,3,4,1])
        self.assert_vec(d, [3,8,9])
        e,l,d = msa_hmm.align.extend_mods(pos_expand=np.array([]), 
                                            expansion_lens=np.array([]), 
                                            pos_discard=np.array([1]),
                                            L=11, k=1)
        self.assert_vec(e, [1])
        self.assert_vec(l, [1])
        self.assert_vec(d, [1,2])
        e,l,d = msa_hmm.align.extend_mods(pos_expand=np.array([5]), 
                                            expansion_lens=np.array([3]), 
                                            pos_discard=np.array(list(range(8))),
                                            L=8)
        self.assert_vec(e, [4])
        self.assert_vec(l, [2])
        self.assert_vec(d, list(range(7)))
        e,l,d = msa_hmm.align.extend_mods(pos_expand=np.array([5]), 
                                            expansion_lens=np.array([3]), 
                                            pos_discard=np.array(list(range(8))),
                                            L=9, k=1)
        self.assert_vec(e, [5])
        self.assert_vec(l, [3])
        self.assert_vec(d, list(range(8)))
        
        e,l,d = msa_hmm.align.extend_mods(pos_expand=np.array([5]), 
                                            expansion_lens=np.array([3]), 
                                            pos_discard=np.array([0,1,2,4,5,6,7]),
                                            L=8)
        self.assert_vec(e, [4])
        self.assert_vec(l, [3])
        self.assert_vec(d, list(range(7)))
        e,l,d = msa_hmm.align.extend_mods(pos_expand=np.array([5]), 
                                            expansion_lens=np.array([3]), 
                                            pos_discard=np.array([0,1,2,4,5,6,7]),
                                            L=9, k=1)
        self.assert_vec(e, [0,5])
        self.assert_vec(l, [1,3])
        self.assert_vec(d, list(range(8)))
        e,l,d = msa_hmm.align.extend_mods(pos_expand=np.array([0,10]), 
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
        length=5
        config = msa_hmm.config.make_default(1)
        emission_init = string_to_one_hot("FELIX").numpy()*20
        insert_init= np.squeeze(string_to_one_hot("A") + string_to_one_hot("B") + string_to_one_hot("C"))*20
        config["emitter"] = msa_hmm.emit.ProfileHMMEmitter(emission_init = tf.constant_initializer(emission_init), 
                                                           insertion_init = tf.constant_initializer(insert_init))
        config["transitioner"] = msa_hmm.trans.ProfileHMMTransitioner(transition_init =(
                            msa_hmm.initializers.make_default_transition_init(MM = 0, 
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
        model = msa_hmm.train.default_model_generator(num_seq=8, 
                                                      effective_num_seq=8,
                                                      model_lengths=[length], 
                                                      config=config)
        #subalignment
        filename = os.path.dirname(__file__)+"/data/felix.fa"
        fasta_file = msa_hmm.fasta.Fasta(filename)
        subset = np.array([0,2,5])
        batch_gen = msa_hmm.train.DefaultBatchGenerator(fasta_file, 1)
        #create alignment after building model
        subalignment = msa_hmm.Alignment(fasta_file, batch_gen, subset, 32, model)
        subalignment_strings = subalignment.to_string(0, add_block_sep=False)
        ref_subalignment = ["FE...LIX...", "FE...LIXbac", "FEabcLIX..."]
        for s,r in zip(subalignment_strings, ref_subalignment):
            self.assertEqual(s,r)
       
    #this test aims to test the high level alignment function by feeding real world data to it
    #and checking if the resulting alignment meets some friendly thresholds 
    def test_alignment_egf(self):
        train_filename = os.path.dirname(__file__)+"/data/egf.fasta"
        ref_filename = os.path.dirname(__file__)+"/data/egf.ref"
        fasta_file = msa_hmm.fasta.Fasta(train_filename)
        ref_file = msa_hmm.fasta.Fasta(ref_filename, aligned=True)
        ref_subset = np.array([fasta_file.seq_ids.index(sid) for sid in ref_file.seq_ids])
        config = msa_hmm.config.make_default(1)
        config["max_surgery_runs"] = 2 #do minimal surgery 
        config["epochs"] = [5,1,5]
        alignment = msa_hmm.align.fit_and_align(fasta_file, 
                                                config=config,
                                                subset=ref_subset, 
                                                verbose=False)
        #some friendly thresholds to check if the alignments does make sense at all
        self.assertTrue(alignment.loglik > -70)
        self.assertTrue(alignment.msa_hmm_layer.cell.length[0] > 25)
        alignment.to_file(os.path.dirname(__file__)+"/data/egf.out.fasta", 0)
        pred_fasta_file = msa_hmm.fasta.Fasta(os.path.dirname(__file__)+"/data/egf.out.fasta")
        p,r = pred_fasta_file.precision_recall(ref_file)
        tc = pred_fasta_file.tc_score(ref_file)
        #based on experience, any half decent hyperparameter choice should yield at least these scores
        self.assertTrue(p > 0.7)
        self.assertTrue(r > 0.7)
        self.assertTrue(tc > 0.1)
        
        
class ConsoleTest(unittest.TestCase):
        
    def test_error_handling(self):
        import subprocess
        
        single_seq = "test/data/single_sequence.fasta"
        faulty_format = "test/data/faulty_format.fasta"
        empty_seq = "test/data/empty_sequence.fasta"
        unknown_symbol = "test/data/unknown_symbol.fasta"
        faulty_msa = "test/data/faulty_msa.fasta"
        
        single_seq_expected_err = f"File {single_seq} contains only a single sequence."
        faulty_format_expected_err = f"Can not read sequences from file {faulty_format}. Expected a file in FASTA format containing at least 2 sequences."
        empty_seq_expected_err = f"File {empty_seq} contains an empty sequence: \'zweite Sequenz\'."
        unknown_symbol_expected_err = f"In file {unknown_symbol}: Found unknown character(s) J in sequence \'erste Sequenz\'. Allowed alphabet: ARNDCQEGHILKMFPSTWYVBZXUO."
        faulty_msa_expected_err = f"In file {faulty_msa}: Although they contain gaps, the input sequences have different lengths. The file seems to contain a malformed alignment."
        
        test = subprocess.Popen(["python", "learnMSA.py", "--silent", "-o", "test.out", "-i", single_seq], stderr=subprocess.PIPE)
        output = test.communicate()[1].strip().decode('ascii')
        self.assertEqual(single_seq_expected_err, output)
        
        test = subprocess.Popen(["python", "learnMSA.py", "--silent", "-o", "test.out", "-i", faulty_format], stderr=subprocess.PIPE)
        output = test.communicate()[1].strip().decode('ascii')
        self.assertEqual(faulty_format_expected_err, output)
        
        test = subprocess.Popen(["python", "learnMSA.py", "--silent", "-o", "test.out", "-i", empty_seq], stderr=subprocess.PIPE)
        output = test.communicate()[1].strip().decode('ascii')
        self.assertEqual(empty_seq_expected_err, output)
        
        test = subprocess.Popen(["python", "learnMSA.py", "--silent", "-o", "test.out", "-i", unknown_symbol], stderr=subprocess.PIPE)
        output = test.communicate()[1].strip().decode('ascii')
        self.assertEqual(unknown_symbol_expected_err, output)
        
        test = subprocess.Popen(["python", "learnMSA.py", "--silent", "-o", "test.out", "-i", faulty_msa], stderr=subprocess.PIPE)
        output = test.communicate()[1].strip().decode('ascii')
        self.assertEqual(faulty_msa_expected_err, output)
        
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
            log_pdf = msa_hmm.dm.dirichlet_log_pdf(probs, alpha, q)
            np.testing.assert_almost_equal(log_pdf, e, decimal=3)
            alpha_init = tf.constant_initializer(msa_hmm.anc_probs.inverse_softplus(alpha))
            mix_init = tf.constant_initializer(np.log(q))
            mean_log_pdf = msa_hmm.dm.DirichletMixtureLayer(1,3,
                                                            alpha_init=alpha_init,
                                                            mix_init=mix_init)(probs)
            np.testing.assert_almost_equal(mean_log_pdf, np.mean(e), decimal=3)
            
    def test_dirichlet_log_pdf_mix(self):
        epsilon = 1e-16
        alpha = np.array([ [1., 1., 1.], [1., 2, 3], [50., 50., 50.], [100., 1., 10.] ])
        probs = np.array([[.2, .3, .5], [1.-2*epsilon, epsilon, epsilon], [.8, .1, .1], [.3, .3, .4]])
        
        expected = np.array([0.48613059, -0.69314836, -0.65780917,  2.1857463])
        q = np.array([0.25, 0.25, 0.25, 0.25])
        log_pdf = msa_hmm.dm.dirichlet_log_pdf(probs, alpha, q)
        np.testing.assert_almost_equal(log_pdf, expected, decimal=3)
        alpha_init = tf.constant_initializer(msa_hmm.anc_probs.inverse_softplus(alpha))
        mix_init = tf.constant_initializer(np.log(q))
        mean_log_pdf = msa_hmm.dm.DirichletMixtureLayer(4, 3,
                                                        alpha_init=alpha_init,
                                                        mix_init=mix_init)(probs)
        np.testing.assert_almost_equal(mean_log_pdf, np.mean(expected), decimal=3)
        
        expected2 = np.array([0.39899244, 0.33647106, 0.33903092, 1.36464418])
        q2 = np.array([0.7, 0.02, 0.08, 0.2])
        log_pdf2 = msa_hmm.dm.dirichlet_log_pdf(probs, alpha, q2)
        np.testing.assert_almost_equal(log_pdf2, expected2, decimal=3)
        mix_init2 = tf.constant_initializer(np.log(q2))
        mean_log_pdf2 = msa_hmm.dm.DirichletMixtureLayer(4, 3,
                                                        alpha_init=alpha_init,
                                                        mix_init=mix_init2)(probs)
        np.testing.assert_almost_equal(mean_log_pdf2, np.mean(expected2), decimal=3)
        
if __name__ == '__main__':
    unittest.main()