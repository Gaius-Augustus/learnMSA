import tensorflow as tf
import numpy as np
import learnMSA.msa_hmm.Utility as ut
import learnMSA.msa_hmm.Fasta as fasta


# terminology
# main model : pHMM core
# flanks (left/right) : unannotated prefixes and suffixes of protein seqs
# unannotated segment : describes the segments between main model matches
# begin/end : special states from which the main model can be entered and exited
# terminal : special state that only emits the terminal-symbol with probability 1 
#
# explicit model : contains all silent states (e.g. deletes)
# implicit model : All paths over silent states are replaced with edges and path probabilities are folded .
#                  In the implicit model all states are emitting and the forward recursion can be
#                  computed by a matrix multiplication.



# MSA HMM Cell based on https://github.com/mslehre/classify-seqs/blob/main/HMMCell.py
class MsaHmmCell(tf.keras.layers.Layer):
    def __init__(self,
                length, # length of the consensus, defines the number of hidden states
                emission_init=None,
                transition_init={}, # a dictionary that maps part_names to arrays of the respective part length
                                        # missing values will default to zero-arrays
                flank_init=0,
                trainable_kernels={} #can be used to omit parameter updates for certain kernels by adding "kernel_id" : False
                ):
        super(MsaHmmCell, self).__init__(dtype=ut.dtype)
        self.length = length 
        #number of explicit (emitting states, i.e. without flanking states and deletions)
        self.num_states = 2 * length + 3 
        self.num_states_implicit = self.num_states + self.length + 2
        self.state_size = (self.num_states, 1)
        self.output_size = self.num_states
        #sub-arrays of the complete transition kernel for convenience
        #describes name and length for kernels of the categorical transition distributions
        #for states or families of states
        self.explicit_transition_kernel_parts = [("begin_to_match", self.length), ("match_to_end", self.length),
                                                ("match_to_match", self.length-1), ("match_to_insert", self.length-1),
                                                ("insert_to_match", self.length-1), ("insert_to_insert", self.length-1),
                                                #consider begin and end states as additional match states:
                                                ("match_to_delete", self.length), ("delete_to_match", self.length),
                                                ("delete_to_delete", self.length-1),
                                                 ("left_flank_loop", 1), ("left_flank_exit", 1),
                                                 ("unannotated_segment_loop", 1), ("unannotated_segment_exit", 1),
                                                 ("right_flank_loop", 1), ("right_flank_exit", 1),
                                                 ("end_to_unannotated_segment", 1), ("end_to_right_flank", 1), ("end_to_terminal", 1)]
        self.implicit_transition_parts = ([("left_flank_loop", 1),
                                           ("left_flank_to_match", self.length),
                                           ("left_flank_to_right_flank", 1),
                                           ("left_flank_to_unannotated_segment", 1),
                                           ("left_flank_to_terminal", 1),
                                           ("match_to_match", self.length-1),
                                           ("match_skip", int((self.length-1) * (self.length-2) / 2)),
                                           ("match_to_unannotated", self.length),
                                           ("match_to_right_flank", self.length),
                                           ("match_to_terminal", self.length),
                                           ("match_to_insert", self.length-1),
                                           ("insert_to_match", self.length-1),
                                           ("insert_to_insert", self.length-1),
                                           ("unannotated_segment_to_match", self.length),
                                           ("unannotated_segment_loop", 1),
                                           ("unannotated_segment_to_right_flank", 1),
                                           ("unannotated_segment_to_terminal", 1),
                                           ("right_flank_loop", 1),
                                           ("right_flank_exit", 1),
                                           ("terminal_self_loop", 1)])
        _assert_transition_init_kernel(transition_init, self.explicit_transition_kernel_parts)
        self.end_state_emission = tf.one_hot([fasta.s-1], fasta.s, dtype=ut.dtype) # end state can only emit the end symbol
        #we add weights only for the match states
        #the end state and all insertion states will emit based on a constant background distribution
        if emission_init is None:
            emission_init = np.zeros((self.length, fasta.s-1))
        self.emission_kernel = self.add_weight(
            shape=(self.length, fasta.s-1), 
            initializer=tf.constant_initializer(emission_init), 
            name='emission_kernel',
            dtype=ut.dtype) # closely related to the emission matrix
        # The (sparse) kernel is subdivided in groups of transitions.
        # To avoid error-prone slicing of a long array into smaller parts,
        # we store the parts as a dictionary and concatenate them later in the correct order.
        # The kernel is closely related to the transition matrix in the implicit model with deletion states.
        self.transition_kernel = {}
        for part_name, length in self.explicit_transition_kernel_parts:
            if part_name == "unannotated_segment_loop" or part_name == "right_flank_loop":
                self.transition_kernel[part_name] = self.transition_kernel["left_flank_loop"]
            elif part_name == "unannotated_segment_exit" or part_name == "right_flank_exit":
                self.transition_kernel[part_name] = self.transition_kernel["left_flank_exit"]
            else:
                self.transition_kernel[part_name] = self.add_weight(shape=[length], 
                                        initializer=tf.constant_initializer(
                                            transition_init.get(part_name, np.zeros(length))),
                                        name="transition_kernel_"+part_name,
                                        trainable=trainable_kernels.get(part_name, True),
                                        dtype=ut.dtype)
        # closely related to the initial probability of the left flank state
        self.flank_init = self.add_weight(shape=(1),
                                         initializer=tf.constant_initializer(flank_init),
                                         name="init_logit",
                                         dtype=ut.dtype)
        
        
        
    #returns 2D indices for the kernel of a sparse (2L+3 x 2L+3) transition matrix without silent states
    #assumes the following ordering of states:
    # [LEFT_FLANK, MATCH x length, INSERT x length-1, UNANNOTATED_SEGMENT, RIGHT_FLANK, TERMINAL]
    def sparse_transition_indices_implicit(self):
        a = np.arange(self.length+1)
        left_flank = 0
        first_insert = self.length + 1
        unanno_segment = 2*self.length
        right_flank = 2*self.length + 1
        terminal = 2*self.length + 2
        indices_dict = {
            "left_flank_loop" : [[left_flank, left_flank]],
            "left_flank_to_match" : np.stack([np.zeros(self.length)+left_flank, a[1:]], axis=1),
            "left_flank_to_right_flank" : [[left_flank, right_flank]],
            "left_flank_to_unannotated_segment" : [[left_flank, unanno_segment]],
            "left_flank_to_terminal" : [[left_flank, terminal]],
            "match_to_match" : np.stack([a[1:-1], a[1:-1]+1], axis=1),
            "match_skip" : np.concatenate([np.stack([np.zeros(self.length-i-1)+i, 
                                         np.arange(i+2, self.length+1)], axis=1)
                for i in range(1, self.length-1)
                    ], axis=0),
            "match_to_unannotated" : np.stack([a[1:], np.zeros(self.length)+unanno_segment], axis=1),
            "match_to_right_flank" : np.stack([a[1:], np.zeros(self.length)+right_flank], axis=1),
            "match_to_terminal" : np.stack([a[1:], np.zeros(self.length)+terminal], axis=1),
            "match_to_insert" : np.stack([a[1:-1], a[:-2]+first_insert], axis=1),
            "insert_to_match" : np.stack([a[:-2]+first_insert, a[2:]], axis=1),
            "insert_to_insert" : np.stack([a[:-2]+first_insert]*2, axis=1),
            "unannotated_segment_to_match" : np.stack([np.zeros(self.length)+unanno_segment, a[1:]], axis=1),
            "unannotated_segment_loop" : [[unanno_segment, unanno_segment]],
            "unannotated_segment_to_right_flank" : [[unanno_segment, right_flank]],
            "unannotated_segment_to_terminal" : [[unanno_segment, terminal]],
            "right_flank_loop" : [[right_flank, right_flank]],
            "right_flank_exit" : [[right_flank, terminal]],
            "terminal_self_loop" : [[terminal, terminal]]}
        return indices_dict


    #returns 2D indices for the (linear) kernel of a sparse (3L+3 x 3L+3) transition matrix with silent states
    #assumes the following ordering of states:
    # [LEFT_FLANK, MATCH x length, INSERT x length-1, UNANNOTATED_SEGMENT, RIGHT_FLANK, TERMINAL, BEGIN, END, DELETE x length]
    def sparse_transition_indices_explicit(self):
        a = np.arange(self.length+1)
        left_flank = 0
        first_insert = self.length + 1
        unanno_segment = 2*self.length
        right_flank = 2*self.length + 1
        terminal = 2*self.length + 2
        begin = 2*self.length + 3
        end = 2*self.length + 4
        first_delete = 2*self.length + 5
        indices_dict = {
            "begin_to_match" : np.stack([np.zeros(self.length)+begin, a[1:]], axis=1),
            "match_to_end" : np.stack([a[1:], np.zeros(self.length)+end], axis=1),
            "match_to_match" : np.stack([a[1:-1], a[1:-1]+1], axis=1),
            "match_to_insert" : np.stack([a[1:-1], a[:-2]+first_insert], axis=1),
            "insert_to_match" : np.stack([a[:-2]+first_insert, a[2:]], axis=1),
            "insert_to_insert" : np.stack([a[:-2]+first_insert]*2, axis=1),
            "match_to_delete" : np.stack([np.insert(a[1:-1], 0, begin), a[:-1]+first_delete], axis=1),
            "delete_to_match" : np.stack([a[:-1]+first_delete, np.append(a[:-2]+2, end)], axis=1),
            "delete_to_delete" : np.stack([a[:-2]+first_delete, a[:-2]+first_delete+1], axis=1),
            "left_flank_loop" : [[left_flank, left_flank]],
            "left_flank_exit" : [[left_flank, begin]],
            "unannotated_segment_loop" : [[unanno_segment, unanno_segment]],
            "unannotated_segment_exit" : [[unanno_segment, begin]],
            "right_flank_loop" : [[right_flank, right_flank]],
            "right_flank_exit" : [[right_flank, terminal]],
            "end_to_unannotated_segment" : [[end, unanno_segment]],
            "end_to_right_flank" : [[end, right_flank]],
            "end_to_terminal" : [[end, terminal]] }
        return indices_dict

        
    def init_cell(self):
        self.A = self.make_A_sparse()
        self.B = self.make_B()
        self.probs = self.make_probs()
        self.init = True
        
        
    def make_transition_kernel(self):
        return tf.concat([self.transition_kernel[part_name] for part_name,_ in self.explicit_transition_kernel_parts], axis=0)
                                
            
    def make_probs(self):
        indices_explicit_dict = self.sparse_transition_indices_explicit()
        indices_explicit = np.concatenate([indices_explicit_dict[part_name] 
                                                for part_name,_ in self.explicit_transition_kernel_parts], axis=0)
        # tf.sparse requires a strict row-major ordering of the indices
        # however, a custom ordering is more convenient in code
        # these indices revert a tf.sparse.reorder:
        a = np.argsort([i*(self.num_states_implicit)+j for i,j in indices_explicit])
        reverse_reorder_indices = np.argsort(a)
        sparse_kernel =  tf.sparse.SparseTensor(
                                    indices=indices_explicit, 
                                    values=self.make_transition_kernel(), 
                                    dense_shape=[self.num_states_implicit]*2)
        sparse_kernel = tf.sparse.reorder(sparse_kernel)
        probs = tf.sparse.softmax(sparse_kernel, name="A") #ignores implicit zeros
        probs_vec = tf.gather(probs.values, reverse_reorder_indices) #revert tf.sparse.reorder
        probs_dict = {}
        lsum = 0
        for part_name, length in self.explicit_transition_kernel_parts:
            probs_dict[part_name] = probs_vec[lsum : lsum+length] 
            lsum += length
        return probs_dict
    
    
    def make_implicit_probs(self):
        probs = self.make_probs()
        #compute match_skip(i,j) = P(Mj+2 | Mi)  , L x L
        #considers "begin" as M0 and "end" as ML
        MD = tf.expand_dims(probs["match_to_delete"], -1)
        DD = tf.concat([[1], probs["delete_to_delete"]], axis=0)
        #compute the cumulative products in log-space to avoid underflow
        DD_cumprod = tf.math.cumprod(DD)
        DD = tf.expand_dims(DD_cumprod, 0) / tf.expand_dims(tf.math.maximum(DD_cumprod, 1e-100), 1)
        DM = tf.expand_dims(probs["delete_to_match"], 0)
        M_skip = MD * DD * DM 
        upper_triangle = tf.linalg.band_part(tf.ones([self.length-2]*2, dtype=ut.dtype), 0, -1)
        entry_add = probs["begin_to_match"] + tf.concat([[0], M_skip[0, :-1]], axis=0)
        exit_add = probs["match_to_end"] + tf.concat([M_skip[1:,-1], [0]], axis=0)
        imp_probs = {}
        imp_probs["match_to_match"] = probs["match_to_match"]
        imp_probs["match_to_insert"] = probs["match_to_insert"]
        imp_probs["insert_to_match"] = probs["insert_to_match"]
        imp_probs["insert_to_insert"] = probs["insert_to_insert"]
        imp_probs["left_flank_loop"] = probs["left_flank_loop"]
        imp_probs["right_flank_loop"] = probs["right_flank_loop"]
        imp_probs["right_flank_exit"] = probs["right_flank_exit"]
        imp_probs["match_skip"] = tf.boolean_mask(M_skip[1:-1, 1:-1], 
                                 mask=tf.cast(upper_triangle, dtype=tf.bool)) 
        imp_probs["left_flank_to_match"] = probs["left_flank_exit"] * entry_add
        imp_probs["left_flank_to_right_flank"] = probs["left_flank_exit"] * M_skip[0, -1] * probs["end_to_right_flank"]
        imp_probs["left_flank_to_unannotated_segment"] = probs["left_flank_exit"] * M_skip[0, -1] * probs["end_to_unannotated_segment"]
        imp_probs["left_flank_to_terminal"] = probs["left_flank_exit"] * M_skip[0, -1] * probs["end_to_terminal"]
        imp_probs["match_to_unannotated"] = exit_add * probs["end_to_unannotated_segment"]
        imp_probs["match_to_right_flank"] = exit_add * probs["end_to_right_flank"]
        imp_probs["match_to_terminal"] = exit_add * probs["end_to_terminal"]
        imp_probs["unannotated_segment_to_match"] = probs["unannotated_segment_exit"] * entry_add
        imp_probs["unannotated_segment_loop"] = probs["unannotated_segment_loop"] + (probs["unannotated_segment_exit"] * M_skip[0, -1] * probs["end_to_unannotated_segment"])
        imp_probs["unannotated_segment_to_right_flank"] = probs["unannotated_segment_exit"] * M_skip[0, -1] * probs["end_to_right_flank"]
        imp_probs["unannotated_segment_to_terminal"] = probs["unannotated_segment_exit"] * M_skip[0, -1] * probs["end_to_terminal"]
        imp_probs["terminal_self_loop"] = tf.ones((1), dtype=ut.dtype)
        return imp_probs
    
    
    def make_A_sparse(self):
        probs = self.make_implicit_probs()
        values = tf.concat([probs[part_name] for part_name,_ in self.implicit_transition_parts], axis=0)
        indices_implicit_dict = self.sparse_transition_indices_implicit()
        indices_implicit = np.concatenate([indices_implicit_dict[part_name] 
                                                for part_name,_ in self.implicit_transition_parts], axis=0)
        A = tf.sparse.SparseTensor(
                            indices=indices_implicit, 
                            values=values, 
                            dense_shape=[self.num_states]*2)
        A = tf.sparse.reorder(A)
        return A
        
        
    def make_A(self):
        A = self.make_A_sparse()
        A = tf.sparse.to_dense(A)
        return A
        
        
    def make_B(self):
        match_emissions = tf.nn.softmax(self.emission_kernel, axis=-1, name="B")
        insertions_emissions = tf.stack([ut.background_distribution]*(self.length+2))
        return  tf.concat([
                      tf.concat([insertions_emissions[:1], 
                                 tf.zeros((1, 1), dtype=ut.dtype)], axis=-1),
                      tf.concat([match_emissions, 
                                 tf.zeros((self.length, 1), dtype=ut.dtype)], axis=-1), 
                      tf.concat([insertions_emissions[1:], 
                                 tf.zeros((self.length+1, 1), dtype=ut.dtype)], axis=-1),
                      self.end_state_emission
                     ], axis=0, name="B")
    
    
    # [LEFT_FLANK, MATCH x length, INSERT x length-1, UNANNOTATED_SEGMENT, RIGHT_FLANK, TERMINAL]
    def make_initial_distribution(self):
        init_flank_prob = tf.math.sigmoid(self.flank_init)
        probs = self.make_probs()
        imp_probs = self.make_implicit_probs()
        init_match = imp_probs["left_flank_to_match"] * (1-init_flank_prob) / probs["left_flank_exit"] 
        init_right_flank = imp_probs["left_flank_to_right_flank"] * (1-init_flank_prob) / probs["left_flank_exit"] 
        init_unannotated_segment = imp_probs["left_flank_to_unannotated_segment"] * (1-init_flank_prob) / probs["left_flank_exit"] 
        init_terminal = imp_probs["left_flank_to_terminal"] * (1-init_flank_prob) / probs["left_flank_exit"] 
        return tf.expand_dims(tf.concat([init_flank_prob, 
                                         init_match, 
                                         tf.zeros(self.length-1, dtype=ut.dtype), 
                                         init_unannotated_segment, init_right_flank, init_terminal], axis=0), 0)

    
    def call(self, inputs, states, training=None):
        old_forward, old_loglik = states
        E = tf.linalg.matvec(self.B, inputs)
        if self.init:
            forward = tf.multiply(E, old_forward, name="forward")
            self.init = False
        else:
            R = tf.sparse.sparse_dense_matmul(old_forward, self.A) 
            forward = tf.multiply(E, R, name="forward")
        S = tf.reduce_sum(forward, axis=-1, keepdims=True, name="loglik")
        loglik = old_loglik + tf.math.log(S) 
        forward = forward / S 
        new_state = [forward, loglik]
        return forward, new_state

    
    def get_initial_state(self, inputs=None, batch_size=None, _dtype=None):
        init_dist = tf.repeat(self.make_initial_distribution(), repeats=batch_size, axis=0)
        loglik = tf.zeros((batch_size, 1), dtype=ut.dtype)
        S = [init_dist, loglik]
        return S
        
        
            
def _assert_transition_init_kernel(kernel_init, parts):
    for part_name in kernel_init.keys():
        assert part_name in [part[0] for part in parts], part_name + " is in the kernel init dict but there is no kernel part matching it. Wrong spelling?"
    for part_name,l in parts:
        if part_name in kernel_init:
            assert kernel_init[part_name].size == l, "\"" + part_name + "\" initializer has length " + str(kernel_init[part_name].size) + " but kernel length is " + str(l)