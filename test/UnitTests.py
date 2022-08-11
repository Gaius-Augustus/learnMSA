import sys 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import unittest
import numpy as np
import tensorflow as tf
from learnMSA import msa_hmm 

class TestMsaHmmCell(unittest.TestCase):
    
   
    def test_A(self):
        length = 4
        transition_init_kernel = {"begin_to_match" : [0.6, 0.1, 0.1, 0.1],
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
                                   "unannotated_segment_loop" : [0.6], 
                                   "unannotated_segment_exit" : [0.4],
                                   "right_flank_loop" : [0.6], 
                                   "right_flank_exit" : [0.4],
                                   "end_to_unannotated_segment" : [0.2], 
                                  "end_to_right_flank" : [0.7], 
                                  "end_to_terminal" : [0.1]}
        transition_init_kernel = {part_name : np.log(p) 
                                  for part_name,p in transition_init_kernel.items()}
        emission_init_kernel = np.zeros((length, msa_hmm.fasta.s-1))
        hmm_cell = msa_hmm.MsaHmmCell(length, emission_init_kernel, transition_init_kernel)
        A = hmm_cell.make_A()
        # [LEFT_FLANK, MATCH x length, INSERT x length-1, UNANNOTATED_SEGMENT, RIGHT_FLANK, TERMINAL]
        A_ref = np.zeros((hmm_cell.num_states, hmm_cell.num_states))
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
        A_ref[8,8] = .6 + .4*.1*.2*.5*.2*.2
        A_ref[8,1] = .4*.6
        A_ref[8,2] = .4*(.1+.1*.8)
        A_ref[8,3] = .4*(.1+.1*.2*.5)
        A_ref[8,4] = .4*(.1+.1*.2*.5*.8)
        A_ref[8,9] = .4*.1*.2*.5*.2*.7
        A_ref[8,10] = .4*.1*.2*.5*.2*.1
        A_ref[9,9] = 0.6
        A_ref[9,10] = 0.4
        A_ref[10,10] = 1
        for i in range(hmm_cell.num_states):
            for j in range(hmm_cell.num_states):
                np.testing.assert_almost_equal(A[i,j], A_ref[i,j], err_msg=str(i)+","+str(j))
        
        imp_probs = hmm_cell.make_implicit_probs()
        for part_name in imp_probs.keys():
            self.assertTrue(part_name in [part[0] for part in hmm_cell.implicit_transition_parts], 
                            part_name + " is in the kernel but not under the expected kernel parts. Wrong spelling?")
        for part_name,l in hmm_cell.implicit_transition_parts:
            if part_name in imp_probs:
                kernel_length = tf.size(imp_probs[part_name]).numpy()
                self.assertTrue(kernel_length == l, 
                                "\"" + part_name + "\" implicit probs array has length " + str(kernel_length) + " but kernel length is " + str(l))
                
                

def string_to_one_hot(s):
    i = [msa_hmm.fasta.alphabet.index(aa) for aa in s]
    return tf.one_hot(i, len(msa_hmm.fasta.alphabet)-1)


def get_all_seqs(fasta_file):
    ds = msa_hmm.train.make_dataset(fasta_file, fasta_file.num_seq, shuffle=False)
    for (seq, std_aa_mask, _), _ in ds:
        return seq.numpy(), std_aa_mask.numpy()

class TestMSAHMM(unittest.TestCase):
    
    def assert_vec(self, x, y):
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(np.all(x == y), str(x) + " not equal to " + str(y))
    
    
    def test_matrices(self):
        length=32
        hmm_cell = msa_hmm.MsaHmmCell(length=length, 
                                      emission_init=msa_hmm.kernel.make_emission_init_kernel(length),
                                      transition_init=msa_hmm.kernel.make_transition_init_kernel(length, **msa_hmm.kernel.transition_init_kernels["default"]))
        A = hmm_cell.make_A()
        A_sum = np.sum(A, -1)
        for a in A_sum:
            np.testing.assert_almost_equal(a, 1.0)
        B = hmm_cell.make_B()
        B_sum = np.sum(B, -1)
        for b in B_sum:
            np.testing.assert_almost_equal(b, 1.0)
            
            
    def test_cell(self):
        length = 4
        emission_init = string_to_one_hot("ACGT").numpy()*10
        transition_init = msa_hmm.kernel.make_transition_init_kernel(length,
                                    MM = 2, 
                                    MI = 0,
                                    MD = 0,
                                    II = 0,
                                    IM = 0,
                                    DM = 0,
                                    DD = 0,
                                    FC = 0,
                                    FE = 3,
                                    R = 0,
                                    RF = 0, 
                                    T = 0)
        hmm_cell = msa_hmm.MsaHmmCell(length=length,  
                                      emission_init=emission_init,
                                      transition_init=transition_init)
        hmm_cell.init_cell()
        fasta_file = msa_hmm.fasta.Fasta(os.path.dirname(__file__)+"/data/simple.fa", gaps = False, contains_lower_case = True)
        sequences, std_aa_mask = get_all_seqs(fasta_file)
        self.assertEqual(sequences.shape, (1,5,len(msa_hmm.fasta.alphabet)))
        forward, loglik = hmm_cell.get_initial_state(batch_size=1)
        self.assertEqual(loglik[0], 0)
        #next match state should always yield highest probability
        for i in range(length):
            _, (forward, loglik) = hmm_cell(sequences[:,i], (forward, loglik))
            self.assertEqual(np.argmax(forward[0]), i+1)
        last_loglik = loglik
        #check correct end in match state
        _, (forward, loglik) = hmm_cell(sequences[:,4], (forward, loglik))
        self.assertEqual(np.argmax(forward[0]), 2*length+2)
        
        hmm_cell.init_cell()
        fasta_file = msa_hmm.fasta.Fasta(os.path.dirname(__file__)+"/data/length_diff.fa", gaps = False, contains_lower_case = True)
        sequences, std_aa_mask = get_all_seqs(fasta_file)
        self.assertEqual(sequences.shape, (2,10,len(msa_hmm.fasta.alphabet)))
        forward, loglik = hmm_cell.get_initial_state(batch_size=1)
        for i in range(length):
            _, (forward, loglik) = hmm_cell(sequences[:,i], (forward, loglik))
            self.assertEqual(np.argmax(forward[0]), i+1)
            self.assertEqual(np.argmax(forward[1]), i+1)
        _, (forward, loglik) = hmm_cell(sequences[:,length], (forward, loglik))
        self.assertEqual(np.argmax(forward[0]), 2*length+2)
        self.assertEqual(np.argmax(forward[1]), 2*length)
        for i in range(4):
            old_loglik = loglik
            _, (forward, loglik) = hmm_cell(sequences[:,length+1+i], (forward, loglik))
            #the first sequence is shorter and padded with end-symbols
            #the first end symbol in each sequence affects the likelihood, but this is the
            #same constant for all sequences in the batch
            #further padding does not affect the likelihood
            self.assertEqual(old_loglik[0,0], loglik[0,0])
            #the second sequence has the motif of the first seq. repeated twice
            #check whether the model loops correctly 
            #looping must yield larger probabilities than using the right flank state
            self.assertEqual(np.argmax(forward[1]), i+1)
            
            
    def test_viterbi(self):
        length = 5
        emission_init = string_to_one_hot("FELIX").numpy()*20
        transition_init = msa_hmm.kernel.make_transition_init_kernel(length,
                                    MM = 0, 
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
                                    T = 0)
        hmm_cell = msa_hmm.MsaHmmCell(length=length, 
                                      emission_init=emission_init,
                                      transition_init=transition_init)
        fasta_file = msa_hmm.fasta.Fasta(os.path.dirname(__file__)+"/data/felix.fa", gaps = False, contains_lower_case = True)
        ref_seqs = np.array([[1,2,3,4,5,12,12,12,12,12,12,12,12,12,12],
                             [0,0,0,1,2,3,4,5,12,12,12,12,12,12,12],
                             [1,2,3,4,5,11,11,11,12,12,12,12,12,12,12],
                             [1,2,3,4,5,10,10,10,1,2,3,4,5,11,12],
                             [0,2,3,4,11,12,12,12,12,12,12,12,12,12,12],
                             [1,2,7,7,7,3,4,5,12,12,12,12,12,12,12],
                             [1,6,6,2,3,8,4,9,9,9,5,12,12,12,12],
                             [1,2,3,8,8,8,4,5,11,11,11,12,12,12,12]])
        sequences, std_aa_mask = get_all_seqs(fasta_file)
        state_seqs_max_lik = msa_hmm.align.viterbi(sequences, hmm_cell)
        # states : [LEFT_FLANK, MATCH x length, INSERT x length-1, UNANNOTATED_SEGMENT, RIGHT_FLANK, END]
        for i in range(fasta_file.num_seq):
            self.assert_vec(state_seqs_max_lik[i], ref_seqs[i])
        #this produces a result identical to above, but runs viterbi batch wise 
        #to avoid memory overflow  
        state_seqs_max_lik2 = msa_hmm.align.get_state_seqs_max_lik(fasta_file,
                                                           np.arange(fasta_file.num_seq),
                                                           batch_size=2,
                                                           msa_hmm_cell=hmm_cell)
        for i in range(fasta_file.num_seq):
            self.assert_vec(state_seqs_max_lik2[i], ref_seqs[i])
        indices = np.array([0,4,5])
        state_seqs_max_lik3 = msa_hmm.align.get_state_seqs_max_lik(fasta_file,
                                                           indices, #try a subset
                                                           batch_size=2,
                                                           msa_hmm_cell=hmm_cell)
        max_len = np.amax(fasta_file.seq_lens[indices])+1
        for i,j in enumerate(indices):
            self.assert_vec(state_seqs_max_lik3[i], ref_seqs[j, :max_len])
            
            
        indices = np.array([0,3,0,0,1,0,0,0]) #skip the left flank 
        decoding_core_results = msa_hmm.align.decode_core(length,
                                                           state_seqs_max_lik,
                                                           indices)
        ref_consensus = np.array([[0,1,2,3,4],
                                  [3,4,5,6,7],
                                  [0,1,2,3,4],
                                  [0,1,2,3,4],
                                  [-1,1,2,3,-1],
                                  [0,1,5,6,7],
                                  [0,3,4,6,10],
                                  [0,1,2,6,7]])
        ref_insertion_lens = np.array([[0]*(length-1),
                                      [0]*(length-1),
                                      [0]*(length-1),
                                      [0]*(length-1),
                                      [0]*(length-1),
                                      [0,3,0,0],
                                      [2,0,1,3],
                                      [0,0,3,0]])
        ref_insertion_start = np.array([[-1]*(length-1),
                                       [-1]*(length-1),
                                       [-1]*(length-1),
                                       [-1]*(length-1),
                                       [-1]*(length-1),
                                       [-1,2,-1,-1],
                                       [1,-1,5,7],
                                       [-1,-1,3,-1]])
        ref_finished = np.array([True, True, True, False, True, True, True, True])
        def assert_decoding_core_results(decoding_core_results, ref):
            for i in range(fasta_file.num_seq):
                self.assert_vec(decoding_core_results[0][i], ref[0][i]) #consensus
                self.assert_vec(decoding_core_results[1][i], ref[1][i]) #ins lens
                self.assert_vec(decoding_core_results[2][i], ref[2][i]) #ins starts
                self.assert_vec(decoding_core_results[3], ref[3]) #finishes
        assert_decoding_core_results(decoding_core_results, (ref_consensus, 
                                                             ref_insertion_lens,
                                                             ref_insertion_start,
                                                             ref_finished)) 
        
        insertion_lens, insertion_start = msa_hmm.align.decode_flank(state_seqs_max_lik, 
                                                                      flank_state_id = 0, 
                                                                      indices = np.array([0,0,0,0,0,0,0,0]))
        self.assert_vec(insertion_lens, np.array([0, 3, 0, 0, 1, 0, 0, 0]))
        self.assert_vec(insertion_start, np.array([0,0,0,0,0,0,0,0]))
        
        core_blocks, left_flank, right_flank, unannotated_segments = msa_hmm.align.decode(length, state_seqs_max_lik)
        self.assertEqual(len(core_blocks), 2)
        assert_decoding_core_results(core_blocks[0], (ref_consensus, 
                                                     ref_insertion_lens,
                                                     ref_insertion_start,
                                                     ref_finished)) 
        ref_consensus_2 = np.array([[-1]*5]*3 + 
                                  [[8,9,10,11,12]] +
                                  [[-1]*5]*4)
        ref_insertion_lens_2 = np.array([[0]*(length-1)]*8)
        ref_insertion_start_2 = np.array([[-1]*(length-1)]*8)
        ref_finished_2 = np.array([True, True, True, True, True, True, True, True])
        assert_decoding_core_results(core_blocks[1], (ref_consensus_2, 
                                                     ref_insertion_lens_2,
                                                     ref_insertion_start_2,
                                                     ref_finished_2))
        self.assert_vec(left_flank[0], np.array([0,3,0,0,1,0,0,0]))
        self.assert_vec(left_flank[1], np.array([0,0,0,0,0,0,0,0]))
        self.assert_vec(unannotated_segments[0][0], np.array([0,0,0,3,0,0,0, 0]))
        self.assert_vec(unannotated_segments[0][1], np.array([5,8,5,5,4,8,11,8]))
        self.assert_vec(right_flank[0], np.array([0,0,3,1, 1,0,0, 3]))
        self.assert_vec(right_flank[1], np.array([5,8,5,13,4,8,11,8]))
        
        sequences_ind = np.argmax(sequences, axis=-1)
        s = msa_hmm.fasta.s
        a = msa_hmm.fasta.alphabet.index("A")+s
        b = msa_hmm.fasta.alphabet.index("B")+s
        c = msa_hmm.fasta.alphabet.index("C")+s
        F = msa_hmm.fasta.alphabet.index("F")
        E = msa_hmm.fasta.alphabet.index("E")
        L = msa_hmm.fasta.alphabet.index("L")
        I = msa_hmm.fasta.alphabet.index("I")
        X = msa_hmm.fasta.alphabet.index("X")
        GAP = s-1
        gap = 2*s-1
        left_flank_block = msa_hmm.align.get_insertion_block(sequences_ind, 
                                                     left_flank[0], 
                                                     np.amax(left_flank[0]),
                                                     left_flank[1],
                                                     align_to_right=True)
        ref_left_flank_block = np.array([[gap]*3, 
                                         [a,b,c],
                                         [gap]*3, 
                                        [gap]*3, 
                                        [gap, gap, a],
                                        [gap]*3, 
                                        [gap]*3, 
                                        [gap]*3])
        self.assert_vec(left_flank_block, ref_left_flank_block)
        right_flank_block = msa_hmm.align.get_insertion_block(sequences_ind, 
                                                             right_flank[0], 
                                                             np.amax(right_flank[0]),
                                                             right_flank[1])
        ref_right_flank_block = np.array([[gap]*3, 
                                          [gap]*3,
                                          [b,a,c], 
                                          [a,gap,gap], 
                                          [b, gap, gap],
                                          [gap]*3, 
                                          [gap]*3, 
                                          [a,b,c]])
        self.assert_vec(right_flank_block, ref_right_flank_block)
        ins_lens = core_blocks[0][1][:,1]
        ins_start = core_blocks[0][2][:,1]
        ins_block = msa_hmm.align.get_insertion_block(sequences_ind, 
                                                      ins_lens, 
                                                      np.amax(ins_lens),
                                                      ins_start)
        ref_ins_block = np.array([[gap]*3, 
                                  [gap]*3, 
                                  [gap]*3, 
                                  [gap]*3, 
                                  [gap]*3, 
                                  [a,b,c],
                                  [gap]*3, 
                                  [gap]*3])
        self.assert_vec(ins_block, ref_ins_block)
        
        ref_core_blocks = [np.array([[F,gap,gap,E,gap,gap,gap,L,gap,gap,gap,I,gap,gap,gap,X],
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
                                    [GAP]*5])]
        for (C,IL,IS,f), ref in zip(core_blocks, ref_core_blocks):
            alignment_block = msa_hmm.align.get_alignment_block(sequences_ind, 
                                                                C,IL,np.amax(IL, axis=0),IS)
            self.assert_vec(alignment_block, ref)
            
                
                
                
                
class TestAncProbs(unittest.TestCase):
    
    def test_anc_probs(self):
        fasta_file = msa_hmm.fasta.Fasta(os.path.dirname(__file__)+"/data/simple.fa", gaps = False, contains_lower_case = True)
        sequences, std_aa_mask = get_all_seqs(fasta_file)
        anc_probs_layer = msa_hmm.AncProbsLayer(num_seqs=sequences.shape[0], tau_init=1.0)
        anc_prob_seqs = anc_probs_layer(sequences, std_aa_mask, [0])
        p = np.zeros(26)
        p[:20] = msa_hmm.ut.read_paml_file()[2]
        #test rate matrix property
        for i in range(anc_probs_layer.Q.shape[0]-1):
            for j in range(anc_probs_layer.Q.shape[0]-1):
                np.testing.assert_almost_equal(anc_probs_layer.Q[i,j] * p[i], 
                                               anc_probs_layer.Q[j,i] * p[j])
        #todo: maybe use known properties of amino acids (e.g. polar, charged, aromatic) to test distributions
        #after some time tau
        
        
        
class TestModelSurgery(unittest.TestCase):
    
    
    def assert_vec(self, x, y):
        if isinstance(x, list):
            x = np.array(x)
        if isinstance(y, list):
            y = np.array(y)
        self.assertEqual(x.shape, y.shape, str(x)+" "+str(y))
        self.assertTrue(np.all(x == y), str(x) + " not equal to " + str(y))
        
    def test_discard_or_expand_positions(self):
        emission_init = string_to_one_hot("FELIC").numpy()*10
        transition_init = msa_hmm.kernel.make_transition_init_kernel(5,
                                    MM = 0, 
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
                                    EN1 = 1,
                                    EN = 0,
                                    EX = 0)
        model,_,_ = msa_hmm.train.make_model(num_seq=10,
                                             effective_num_seq=10, 
                                             model_length=5, 
                                             emission_init=emission_init,
                                             transition_init=transition_init,
                                             flank_init="default",
                                             alpha_flank=1e3, 
                                             alpha_single=1e9, 
                                             alpha_frag=1e3,
                                             use_anc_probs=False)
        fasta_file = msa_hmm.fasta.Fasta(os.path.dirname(__file__)+"/data/felix_insert_delete.fa", gaps = False, contains_lower_case = True)
        alignment = msa_hmm.Alignment(fasta_file, 
                                      np.arange(fasta_file.num_seq),
                                      32, 
                                      model,
                                      use_anc_probs=False)
        #a simple alignment to test detection of
        #sparse and unconserved columns and frequent or very long insertions
        ref_seqs = [
            "F.-..........LnnnI-aaaFELnICnnn",
            "-.EnnnnnnnnnnLnn.I-aaaFE-.ICnnn",
            "-.-..........Lnn.I-...---.--nnn",
            "-.-..........-...ICaaaF--.I-nnn",
            "FnE..........-...ICaaaF-LnI-nnn"
        ]
        aligned_sequences = alignment.to_string(add_block_sep=False)
        for s, ref_s in zip(aligned_sequences, ref_seqs):
            self.assertEqual(s, ref_s)
        #shape: [number of domain hits, length]
        deletions = np.sum(alignment.consensus == -1, axis=1)
        self.assert_vec(deletions, [[3,3,2,0,3], [1,3,3,1,3]]) 
        #shape: [number of domain hits, num seq]
        self.assert_vec(alignment.finished, [[False,False,True,False,False], [True,True,True,True,True]]) 
        #shape: [number of domain hits, num seq, L-1 inner positions]
        self.assert_vec(alignment.insertion_lens, [[[0, 0, 3, 0],
                                                  [0, 10, 2, 0],
                                                  [0, 0, 2, 0],
                                                  [0, 0, 0, 0],
                                                  [1, 0, 0, 0]],

                                                 [[0, 0, 1, 0],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 1, 0]]]) 
        pos_expand, expansion_lens, pos_discard = msa_hmm.align.get_discard_or_expand_positions(alignment, ins_long=9, k=1)
        self.assert_vec(pos_expand, [2,3,5])
        self.assert_vec(expansion_lens, [9,1,3])
        self.assert_vec(pos_discard, [4])
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
        emission_init2 = string_to_one_hot("A").numpy()*10
        transition_init2 = msa_hmm.kernel.make_transition_init_kernel(5,
                                    MM = 77, 
                                    MI = 77,
                                    MD = 77,
                                    II = 77,
                                    IM = 77,
                                    DM = 77,
                                    DD = 77,
                                    FC = 77,
                                    FE = 77,
                                    R = 77,
                                    RF = 77, 
                                    T = 77,
                                    EN1 = 77,
                                    EN = 77,
                                    EX = 77)
        transitions_new, emissions_new,_ = msa_hmm.align.update_kernels(alignment, 
                                                              pos_expand, expansion_lens, pos_discard,
                                                              emission_init2, transition_init2, 0.0)
        ref_consensus = "FE"+"A"*9+"LAI"+"A"*3
        self.assertEqual(emissions_new.shape[0], len(ref_consensus))
        self.assert_vec(emissions_new, string_to_one_hot(ref_consensus).numpy()*10)
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
        
        

class TestAlignment(unittest.TestCase):
    
    def test_subalignment(self):
        length=5
        emission_init = string_to_one_hot("FELIX").numpy()*20
        transition_init = msa_hmm.kernel.make_transition_init_kernel(length,
                                    MM = 0, 
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
                                    T = 0)
        model,_,_ = msa_hmm.train.make_model(num_seq=8, 
                                             effective_num_seq=8,
                                             model_length=length, 
                                             emission_init=emission_init,
                                             transition_init=transition_init,
                                             flank_init="default",
                                             alpha_flank=1e3, 
                                             alpha_single=1e9, 
                                             alpha_frag=1e3,
                                             use_anc_probs=False)
    
        #subalignment
        fasta_file = msa_hmm.fasta.Fasta(os.path.dirname(__file__)+"/data/felix.fa", gaps=False, contains_lower_case=True)
        subset = np.array([0,2,5])
        subalignment = msa_hmm.Alignment(fasta_file, subset, 32, model, use_anc_probs=False)
        subalignment_strings = subalignment.to_string(add_block_sep=False)
        ref_subalignment = ["FE...LIX...", "FE...LIXbac", "FEabcLIX..."]
        for s,r in zip(subalignment_strings, ref_subalignment):
            self.assertEqual(s,r)
       
    #this test aims to test the high level alignment function by feeding real world data to it
    #and checking if the resulting alignment meets some friendly thresholds 
    def test_alignment_egf(self):
        fasta_file = msa_hmm.fasta.Fasta(os.path.dirname(__file__)+"/data/egf.fasta", gaps=False, contains_lower_case=True)
        ref_file = msa_hmm.fasta.Fasta(os.path.dirname(__file__)+"/data/egf.ref", gaps=True, contains_lower_case=True)
        ref_subset = np.array([fasta_file.seq_ids.index(sid) for sid in ref_file.seq_ids])
        loglik, alignment = msa_hmm.align.fit_and_align_n(fasta_file, 
                                                          num_runs=1, 
                                                          config=msa_hmm.config.default,
                                                          subset=ref_subset, 
                                                          verbose=False)[0]
        #some friendly thresholds to check if the alignments does make sense at all
        self.assertTrue(loglik > -70)
        self.assertTrue(alignment.msa_hmm_layer.length > 25)
        alignment.to_file(os.path.dirname(__file__)+"/data/egf.out.fasta")
        pred_fasta_file = msa_hmm.fasta.Fasta(os.path.dirname(__file__)+"/data/egf.out.fasta", gaps=True, contains_lower_case=True)
        p,r = pred_fasta_file.precision_recall(ref_file)
        tc = pred_fasta_file.tc_score(ref_file)
        #based on experience, any half decent hyperparameter choice should yield at least these scores
        self.assertTrue(p > 0.7)
        self.assertTrue(r > 0.7)
        self.assertTrue(tc > 0.1)
        
        
if __name__ == '__main__':
    unittest.main()