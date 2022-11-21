import os
import tensorflow as tf
import numpy as np
import learnMSA.msa_hmm.DirichletMixture as dm


class AminoAcidPrior():
    """The default dirichlet mixture prior for amino acids.

    Args:
        comp_count: The number of components in the mixture.
    """
    def __init__(self, comp_count=1):
        self.comp_count = comp_count
        
    def load(self, dtype):
        prior_path = os.path.dirname(__file__)+"/trained_prior/"
        dtype = dtype if isinstance(dtype, str) else dtype.name
        model_path = prior_path+"_".join([str(self.comp_count), "True", dtype, "_dirichlet/ckpt"])
        model = dm.load_mixture_model(model_path, self.comp_count, 20, trainable=False, dtype=dtype)
        self.emission_dirichlet_mix = model.layers[-1]
        
    def __call__(self, B):
        """Computes log pdf values for each match state.
        Args:
        B: The emission matrix. Assumes that the 20 std amino acids are the first 20 positions of dim 2 
            and that the states are ordered the standard way as seen in MsaHmmCell. Shape: (q, s)
        Returns:
        A tensor with the log pdf values of this Dirichlet mixture. Shape: (q)
        """
        model_length = int((tf.shape(B)[0]-2)/2)
        B = B[1:model_length+1,:20]
        B /= tf.reduce_sum(B, axis=-1, keepdims=True)
        prior = self.emission_dirichlet_mix.log_pdf(B)
        return prior 
    
    
class NullPrior():
    """NullObject if no prior should be used.
    """
    def load(self, dtype):
        pass
    
    def __call__(self, B):
        model_length = int((tf.shape(B)[0]-2)/2)
        return tf.zeros((model_length), dtype=B.dtype)
    

def make_default_emission_matrix(s, em, ins, length, dtype):
    """Constructs an emission matrix compatible with MsaHmmCell from kernels.
    Args:
       s: Alphabet size.
       em: Emission distribution logits (kernel).
       ins: Shared insertion distribution (kernel).
       length: Model length.
       dtype: Datatype of the MsaHmmCell.
    Returns:
        A tensor with the log pdf values of this Dirichlet mixture. Shape: (q)
    """
    emissions = tf.concat([tf.expand_dims(ins, 0), 
                           em, 
                           tf.stack([ins]*(length+1))] , axis=0)
    emissions = tf.nn.softmax(emissions)
    emissions = tf.concat([emissions, tf.zeros_like(emissions[:,:1])], axis=-1) 
    end_state_emission = tf.one_hot([s], s+1, dtype=dtype) 
    emissions = tf.concat([emissions, end_state_emission], axis=0)
    return emissions




class MsaHmmCell(tf.keras.layers.Layer):
    """ This class is the core of the learnMSA model. It is meant to be used with the generic RNN-layer.
        It computes the likelihood of a batch of sequences, computes a prior value and provides functionality to
        construct the emission- and transition-matricies also used elsewhere e.g. during Viterbi.
        Based on https://github.com/mslehre/classify-seqs/blob/main/HMMCell.py.
    Args:
        length: Model length / number of match states.
        TODO
        dtype: The datatype of the underlying model.
    """
    def __init__(self,
                length, 
                kernel_dim=25, 
                emission_init = tf.keras.initializers.Zeros(),
                transition_init={"begin_to_match" : tf.keras.initializers.Zeros(), 
                                 "match_to_end" : tf.keras.initializers.Zeros(),
                                 "match_to_match" : tf.keras.initializers.Zeros(), 
                                 "match_to_insert" : tf.keras.initializers.Zeros(),
                                 "insert_to_match" : tf.keras.initializers.Zeros(), 
                                 "insert_to_insert" : tf.keras.initializers.Zeros(),
                                 "match_to_delete" : tf.keras.initializers.Zeros(), 
                                 "delete_to_match" : tf.keras.initializers.Zeros(),
                                 "delete_to_delete" : tf.keras.initializers.Zeros(),
                                 "left_flank_loop" : tf.keras.initializers.Zeros(), 
                                 "left_flank_exit" : tf.keras.initializers.Zeros(),
                                 "unannotated_segment_loop" : tf.keras.initializers.Zeros(), 
                                 "unannotated_segment_exit" : tf.keras.initializers.Zeros(),
                                 "right_flank_loop" : tf.keras.initializers.Zeros(), 
                                 "right_flank_exit" : tf.keras.initializers.Zeros(),
                                 "end_to_unannotated_segment" : tf.keras.initializers.Zeros(), 
                                 "end_to_right_flank" : tf.keras.initializers.Zeros(), 
                                 "end_to_terminal" : tf.keras.initializers.Zeros()},
                insertion_init = tf.keras.initializers.Zeros(),
                flank_init = tf.keras.initializers.Zeros(),
                 
                 #function of form f(B, inputs) that defines how the emission probabilities are computed
                 #per default, categorical dists are assumed
                emission_func = tf.linalg.matvec, 
                emission_matrix_generator = make_default_emission_matrix,
                emission_prior = AminoAcidPrior(),
                alpha_flank = 7e3,
                alpha_single = 1e9,
                alpha_frag = 1e4,
                 
                 #can be used to omit parameter updates for certain kernels by adding "kernel_id" : False
                frozen_kernels={}, 
                frozen_insertions=True,
                dtype=tf.float32,
                **kwargs
                ):
        super(MsaHmmCell, self).__init__(name="MsaHmmCell", dtype=dtype, **kwargs)
        self.length = length 
        self.kernel_dim = kernel_dim
        self.alpha_flank = alpha_flank
        self.alpha_single = alpha_single
        self.alpha_frag = alpha_frag
        self.emission_init = emission_init
        self.transition_init = transition_init
        self.insertion_init = insertion_init
        self.flank_init = flank_init
        self.emission_func = emission_func
        self.emission_matrix_generator = emission_matrix_generator
        self.emission_prior = emission_prior
        self.emission_prior.load(dtype)
        #number of explicit (emitting states, i.e. without flanking states and deletions)
        self.num_states = 2 * length + 3 
        self.num_states_implicit = self.num_states + self.length + 2
        self.state_size = (self.num_states, 1)
        self.output_size = self.num_states
        self.frozen_kernels = frozen_kernels
        self.frozen_insertions = frozen_insertions
        #sub-arrays of the complete transition kernel for convenience
        #describes name and length for kernels of the categorical transition distributions
        #for states or families of states
        self.explicit_transition_kernel_parts = [("begin_to_match", self.length), 
                                                 ("match_to_end", self.length),
                                                 ("match_to_match", self.length-1), 
                                                 ("match_to_insert", self.length-1),
                                                 ("insert_to_match", self.length-1), 
                                                 ("insert_to_insert", self.length-1),
                                                #consider begin and end states as additional match states:
                                                 ("match_to_delete", self.length), 
                                                 ("delete_to_match", self.length),
                                                 ("delete_to_delete", self.length-1),
                                                 ("left_flank_loop", 1), 
                                                 ("left_flank_exit", 1),
                                                 ("unannotated_segment_loop", 1), 
                                                 ("unannotated_segment_exit", 1),
                                                 ("right_flank_loop", 1), 
                                                 ("right_flank_exit", 1),
                                                 ("end_to_unannotated_segment", 1), 
                                                 ("end_to_right_flank", 1), 
                                                 ("end_to_terminal", 1)]
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
        
        _assert_transition_init_kernel(self.transition_init, self.explicit_transition_kernel_parts)
        
        # support multiple emissions
        if not hasattr(self.emission_init, '__iter__'):
            self.emission_init = [self.emission_init]
        if not hasattr(self.insertion_init, '__iter__'):
            self.insertion_init = [self.insertion_init]
        if not hasattr(self.kernel_dim, '__iter__'):
            self.kernel_dim = [self.kernel_dim]
#         if not hasattr(self.emission_func, '__iter__'):
#             self.emission_func = [self.emission_func]
        if not hasattr(self.emission_matrix_generator, '__iter__'):
            self.emission_matrix_generator = [self.emission_matrix_generator]
        if not hasattr(self.emission_prior, '__iter__'):
            self.emission_prior = [self.emission_prior]
        if not hasattr(self.frozen_insertions, '__iter__'):
            self.frozen_insertions = [self.frozen_insertions]
            
        assert all(len(self.emission_init) == len(x) for x in [self.emission_init, self.insertion_init, self.kernel_dim, self.emission_matrix_generator, self.emission_prior, self.frozen_insertions]), "emission_init, insertion_init, kernel_dim, emission_matrix_generator, emission_prior, frozen_insertions must have all the same length" 
            
        self.load_priors()
        
        self.epsilon = tf.constant(1e-32, dtype)
        self.approx_log_zero = tf.math.log(self.epsilon)
            
            
    def build(self, input_shape=None):
        # The (sparse) kernel is subdivided in groups of transitions.
        # To avoid error-prone slicing of a long array into smaller parts,
        # we store the parts as a dictionary and concatenate them later in the correct order.
        # The kernel is closely related to the transition matrix in the implicit model with deletion states.
        self.transition_kernel = {}
        for part_name, length in self.explicit_transition_kernel_parts:
            if part_name == "right_flank_loop": #tied flanks
                self.transition_kernel[part_name] = self.transition_kernel["left_flank_loop"]
            elif part_name == "right_flank_exit": #tied flanks
                self.transition_kernel[part_name] = self.transition_kernel["left_flank_exit"]
            else:
                self.transition_kernel[part_name] = self.add_weight(shape=[length], 
                                        initializer = self.transition_init[part_name],
                                        name="transition_kernel_"+part_name,
                                        trainable=not self.frozen_kernels.get(part_name, False),
                                        dtype=self.dtype)
                
        # closely related to the emission matrix of the match states
        self.emission_kernel = [self.add_weight(
                                        shape=[self.length, s], 
                                        initializer=init, 
                                        name="emission_kernel"+str(i),
                                        dtype=self.dtype) 
                                           for i, (init, s) in enumerate(zip(self.emission_init, self.kernel_dim))]
        
        self.insertion_kernel = [self.add_weight(
                                shape=[s],
                                initializer=init,
                                name="insertion_kernel"+str(i),
                                trainable=not frozen,
                                dtype=self.dtype) 
                                 for i, (init, s, frozen) in enumerate(zip(self.insertion_init, self.kernel_dim, self.frozen_insertions))]
        
        # closely related to the initial probability of the left flank state
        self.flank_init_kernel = self.add_weight(shape=[1],
                                         initializer=self.flank_init,
                                         name="init_logit",
                                         dtype=self.dtype)
        
        
        
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
    # [LEFT_FLANK, MATCH x length, INSERT x length-1, UNANNOTATED_SEGMENT, 
    #  RIGHT_FLANK, TERMINAL, BEGIN, END, DELETE x length]
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
        return tf.concat([self.transition_kernel[part_name] 
                          for part_name,_ in self.explicit_transition_kernel_parts], axis=0)
              
    def make_probs(self):
        indices_explicit_dict = self.sparse_transition_indices_explicit()
        indices_explicit = np.concatenate([indices_explicit_dict[part_name] 
                                                for part_name,_ in 
                                                   self.explicit_transition_kernel_parts], axis=0)
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
    
    def make_log_probs(self):
        return {key : tf.math.log(p) for key,p in self.make_probs().items()}
    
    def make_implicit_log_probs(self):
        log_probs = self.make_log_probs()
        #compute match_skip(i,j) = P(Mj+2 | Mi)  , L x L
        #considers "begin" as M0 and "end" as ML
        MD = tf.expand_dims(log_probs["match_to_delete"], -1)
        DD = tf.concat([[0], log_probs["delete_to_delete"]], axis=0)
        DD_cumsum = tf.math.cumsum(DD)
        DD = tf.expand_dims(DD_cumsum, 0) - tf.expand_dims(DD_cumsum, 1)
        DM = tf.expand_dims(log_probs["delete_to_match"], 0)
        M_skip = MD + DD + DM 
        upper_triangle = tf.linalg.band_part(tf.ones([self.length-2]*2, dtype=self.dtype), 0, -1)
        entry_add = _logsumexp(log_probs["begin_to_match"], 
                               tf.concat([[self.approx_log_zero], M_skip[0, :-1]], axis=0))
        exit_add = _logsumexp(log_probs["match_to_end"], 
                              tf.concat([M_skip[1:,-1], [self.approx_log_zero]], axis=0))
        imp_probs = {}
        imp_probs["match_to_match"] = log_probs["match_to_match"]
        imp_probs["match_to_insert"] = log_probs["match_to_insert"]
        imp_probs["insert_to_match"] = log_probs["insert_to_match"]
        imp_probs["insert_to_insert"] = log_probs["insert_to_insert"]
        imp_probs["left_flank_loop"] = log_probs["left_flank_loop"]
        imp_probs["right_flank_loop"] = log_probs["right_flank_loop"]
        imp_probs["right_flank_exit"] = log_probs["right_flank_exit"]
        imp_probs["match_skip"] = tf.boolean_mask(M_skip[1:-1, 1:-1], 
                                 mask=tf.cast(upper_triangle, dtype=tf.bool)) 
        imp_probs["left_flank_to_match"] = log_probs["left_flank_exit"] + entry_add
        imp_probs["left_flank_to_right_flank"] = log_probs["left_flank_exit"] + M_skip[0, -1] + log_probs["end_to_right_flank"]
        imp_probs["left_flank_to_unannotated_segment"] = log_probs["left_flank_exit"] + M_skip[0, -1] + log_probs["end_to_unannotated_segment"]
        imp_probs["left_flank_to_terminal"] = log_probs["left_flank_exit"] + M_skip[0, -1] + log_probs["end_to_terminal"]
        imp_probs["match_to_unannotated"] = exit_add + log_probs["end_to_unannotated_segment"]
        imp_probs["match_to_right_flank"] = exit_add + log_probs["end_to_right_flank"]
        imp_probs["match_to_terminal"] = exit_add + log_probs["end_to_terminal"]
        imp_probs["unannotated_segment_to_match"] = log_probs["unannotated_segment_exit"] + entry_add
        imp_probs["unannotated_segment_loop"] = _logsumexp(log_probs["unannotated_segment_loop"], 
                                                           (log_probs["unannotated_segment_exit"] 
                                                                + M_skip[0, -1] 
                                                                + log_probs["end_to_unannotated_segment"]))
        imp_probs["unannotated_segment_to_right_flank"] = (log_probs["unannotated_segment_exit"] 
                                                           + M_skip[0, -1] 
                                                           + log_probs["end_to_right_flank"])
        imp_probs["unannotated_segment_to_terminal"] = (log_probs["unannotated_segment_exit"] 
                                                        + M_skip[0, -1] 
                                                        + log_probs["end_to_terminal"])
        imp_probs["terminal_self_loop"] = tf.zeros((1), dtype=self.dtype)
        return imp_probs
    
    
    def make_log_A_sparse(self):
        log_probs = self.make_implicit_log_probs()
        values = tf.concat([log_probs[part_name] for part_name,_ in self.implicit_transition_parts], axis=0)
        indices_implicit_dict = self.sparse_transition_indices_implicit()
        indices_implicit = np.concatenate([indices_implicit_dict[part_name] 
                                                for part_name,_ in self.implicit_transition_parts], axis=0)
        A = tf.sparse.SparseTensor(
                            indices=indices_implicit, 
                            values=values, 
                            dense_shape=[self.num_states]*2)
        A = tf.sparse.reorder(A)
        return A
    
    def make_log_A(self):
        log_A = self.make_log_A_sparse()
        log_A = tf.sparse.to_dense(log_A, default_value=self.approx_log_zero)
        return log_A
    
    def make_implicit_probs(self):
        probs = self.make_probs()
        #compute match_skip(i,j) = P(Mj+2 | Mi)  , L x L
        #considers "begin" as M0 and "end" as ML
        MD = tf.expand_dims(probs["match_to_delete"], -1)
        DD = tf.concat([[1], probs["delete_to_delete"]], axis=0)
        #TODO: compute the cumulative products in log-space to avoid underflow
        DD_cumprod = tf.math.cumprod(DD)
        DD = tf.expand_dims(DD_cumprod, 0) / tf.expand_dims(tf.math.maximum(DD_cumprod, self.epsilon), 1)
        DM = tf.expand_dims(probs["delete_to_match"], 0)
        M_skip = MD * DD * DM 
        upper_triangle = tf.linalg.band_part(tf.ones([self.length-2]*2, dtype=self.dtype), 0, -1)
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
        imp_probs["terminal_self_loop"] = tf.ones((1), dtype=self.dtype)
        return imp_probs
    
    def make_A_sparse(self):
        probs = self.make_implicit_log_probs()
        values = tf.concat([probs[part_name] for part_name,_ in self.implicit_transition_parts], axis=0)
        values = tf.math.exp(values)
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
        B = []
        for gen, s, em, ins in zip(self.emission_matrix_generator, self.kernel_dim, self.emission_kernel, self.insertion_kernel):
            B.append(gen(s, em, ins, self.length, self.dtype))
        return B
    
    
    # [LEFT_FLANK, MATCH x length, INSERT x length-1, UNANNOTATED_SEGMENT, RIGHT_FLANK, TERMINAL]
    def make_initial_distribution(self):
        init_flank_prob = tf.math.sigmoid(self.flank_init_kernel)
        log_init_flank_prob = tf.math.log(init_flank_prob)
        log_complement_init_flank_prob = tf.math.log(1-init_flank_prob)
        log_probs = self.make_log_probs()
        log_imp_probs = self.make_implicit_log_probs()
        log_init_match = (log_imp_probs["left_flank_to_match"] 
                      + log_complement_init_flank_prob 
                      - log_probs["left_flank_exit"])
        log_init_right_flank = (log_imp_probs["left_flank_to_right_flank"] 
                            + log_complement_init_flank_prob 
                            - log_probs["left_flank_exit"])
        log_init_unannotated_segment = (log_imp_probs["left_flank_to_unannotated_segment"] 
                                    + log_complement_init_flank_prob 
                                    - log_probs["left_flank_exit"])
        log_init_terminal = (log_imp_probs["left_flank_to_terminal"] 
                         + log_complement_init_flank_prob 
                         - log_probs["left_flank_exit"] )
        log_init_dist = tf.expand_dims(tf.concat([log_init_flank_prob, 
                                                    log_init_match, 
                                                    tf.zeros(self.length-1, dtype=self.dtype) + self.approx_log_zero, 
                                                    log_init_unannotated_segment, 
                                                    log_init_right_flank, 
                                                    log_init_terminal], axis=0), 0)
        init_dist = tf.math.exp(log_init_dist)
        return init_dist
    
    def emission_probs(self, inputs):
        return self.emission_func(self.B, inputs)

    def call(self, inputs, states, training=None):
        old_forward, old_loglik = states
        E = self.emission_probs(inputs)
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
        loglik = tf.zeros((batch_size, 1), dtype=self.dtype)
        S = [init_dist, loglik]
        return S
    
    
    def load_priors(self, match_comp=1, insert_comp=1, delete_comp=1):
        #equivalent to the alpha parameters of a dirichlet mixture -1 .
        #these values are crutial when jointly optimizing the main model with the additional
        #"Plan7" states and transitions
        #favors high probability of staying one of the 3 flanking states
        self.alpha_flank = tf.constant(self.alpha_flank, dtype=self.dtype) 
        #favors high probability for a single main model hit (avoid looping paths)
        self.alpha_single = tf.constant(self.alpha_single, dtype=self.dtype) 
        #favors models with high prob. to enter at the first match and exit after the last match
        self.alpha_frag = tf.constant(self.alpha_frag, dtype=self.dtype) 
        
        prior_path = os.path.dirname(__file__)+"/trained_prior/transition_priors/"
        
        match_model_path = prior_path + "_".join(["match_prior", str(match_comp), self.dtype]) + "/ckpt"
        match_model = dm.load_mixture_model(match_model_path, match_comp, 3, trainable=False, dtype=self.dtype)
        self.match_dirichlet = match_model.layers[-1]
        
        insert_model_path = prior_path + "_".join(["insert_prior", str(insert_comp), self.dtype]) + "/ckpt"
        insert_model = dm.load_mixture_model(insert_model_path, insert_comp, 2, trainable=False, dtype=self.dtype)
        self.insert_dirichlet = insert_model.layers[-1]
        
        delete_model_path = prior_path + "_".join(["delete_prior", str(delete_comp), self.dtype]) + "/ckpt"
        delete_model = dm.load_mixture_model(delete_model_path, delete_comp, 2, trainable=False, dtype=self.dtype)
        self.delete_dirichlet = delete_model.layers[-1]
        
#         model, DMP = dm.make_model(1, 3, -1, trainable=False)
#         model.load_weights(
#             PRIOR_PATH+"transition_priors/match/ckpt").expect_partial()
#         self.match_dirichlet = dm.DirichletMixturePrior(1, 3, -1,
#                                             DMP.alpha_kernel.numpy(),
#                                             DMP.q_kernel.numpy(),
#                                             trainable=False)
        
#         model, DMP = dm.make_model(1, 2, -1, trainable=False)
#         model.load_weights(
#             PRIOR_PATH+"transition_priors/insert/ckpt").expect_partial()
#         self.insert_dirichlet = dm.DirichletMixturePrior(1, 2, -1,
#                                             DMP.alpha_kernel.numpy(),
#                                             DMP.q_kernel.numpy(),
#                                             trainable=False)
        
#         model, DMP = dm.make_model(1, 2, -1, trainable=False)
#         model.load_weights(
#             PRIOR_PATH+"transition_priors/delete/ckpt").expect_partial()
#         self.delete_dirichlet = dm.DirichletMixturePrior(1, 2, -1,
#                                             DMP.alpha_kernel.numpy(),
#                                             DMP.q_kernel.numpy(),
#                                             trainable=False) 
    
    
    def get_prior_log_density(self, add_metrics=False):    
        probs = self.make_probs()
        log_probs = {part_name : tf.math.log(p) for part_name, p in probs.items()}
        #emissions
        em_dirichlets = [tf.reduce_sum(em_prior(emission)) for emission, em_prior in zip(self.B, self.emission_prior)]
        #match state transitions
        p_match = tf.stack([probs["match_to_match"],
                            probs["match_to_insert"],
                            probs["match_to_delete"][1:]], axis=-1)
        p_match_sum = tf.reduce_sum(p_match, axis=-1, keepdims=True)
        match_dirichlet = tf.reduce_sum(self.match_dirichlet.log_pdf(p_match / p_match_sum))
        #insert state transitions
        p_insert = tf.stack([probs["insert_to_match"],
                           probs["insert_to_insert"]], axis=-1)
        insert_dirichlet = tf.reduce_sum(self.insert_dirichlet.log_pdf(p_insert))
        #delete state transitions
        p_delete = tf.stack([probs["delete_to_match"][:-1],
                           probs["delete_to_delete"]], axis=-1)
        delete_dirichlet = tf.reduce_sum(self.delete_dirichlet.log_pdf(p_delete))
        #other transitions
        flank_prior = self.alpha_flank * log_probs["unannotated_segment_loop"] #todo: handle as extra case?
        flank_prior += self.alpha_flank * log_probs["right_flank_loop"]
        flank_prior += self.alpha_flank * log_probs["left_flank_loop"]
        flank_prior += self.alpha_flank * tf.math.log(probs["end_to_right_flank"])
        flank_prior += self.alpha_flank * tf.math.log(tf.math.sigmoid(self.flank_init_kernel))
        #uni-hit
        hit_prior = self.alpha_single * tf.math.log(probs["end_to_right_flank"] + probs["end_to_terminal"])
        #uniform entry/exit prior
        btm = probs["begin_to_match"] / (1- probs["match_to_delete"][0])
        enex_prior = tf.expand_dims(btm, 1) * tf.expand_dims(probs["match_to_end"], 0)
        enex_prior = tf.linalg.band_part(enex_prior, 0, -1)
        enex_prior = tf.math.log(1 - enex_prior) 
        enex_prior = self.alpha_frag * (tf.reduce_sum(enex_prior) - enex_prior[0, -1])
        prior =(sum(em_dirichlets) +
                match_dirichlet +
                insert_dirichlet +
                delete_dirichlet +
                flank_prior +
                hit_prior +
                enex_prior)
        if add_metrics:
            for i,d in enumerate(em_dirichlets):
                self.add_metric(d, "em_dirichlets_"+str(i))
            self.add_metric(match_dirichlet, "match_dirichlet")
            self.add_metric(insert_dirichlet, "insert_dirichlet")
            self.add_metric(delete_dirichlet, "delete_dirichlet")
            self.add_metric(flank_prior, "flank_prior")
            self.add_metric(hit_prior, "hit_prior")
            self.add_metric(enex_prior, "enex_prior")
        prior = tf.cast(prior, self.dtype)
        return prior
        
      
def _logsumexp(x, y):
    return tf.math.log(tf.math.exp(x) +  tf.math.exp(y))
            
def _assert_transition_init_kernel(kernel_init, parts):
    for part_name,_ in parts:
        assert part_name in kernel_init, "No initializer found for kernel " + part_name + "."
    for part_name in kernel_init.keys():
        assert part_name in [part[0] for part in parts], part_name + " is in the kernel init dict but there is no kernel part matching it. Wrong spelling?"