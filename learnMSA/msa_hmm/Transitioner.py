import tensorflow as tf
import numpy as np
import learnMSA.msa_hmm.Initializers as initializers
import learnMSA.msa_hmm.Priors as priors


class ProfileHMMTransitioner(tf.keras.layers.Layer):
    """ A transitioner defines which transitions between HMM states are allowed, how they are initialized
        and how the transition matrix is represented (dense, sparse, other).
        The transitioner also holds a prior on the transition distributions. 
        This transitioner implements the default profile HMM logic with the additional Plan7 states.
    Args:
        transition_init: A dictionary with initializers for each edge type. 
        frozer_kernels: A dictionary that can be used to omit parameter updates for certain kernels 
                        by adding "kernel_id" : False
    """
    def __init__(self, 
                transition_init = initializers.make_default_transition_init(),
                flank_init = initializers.make_default_flank_init(),
                prior = priors.ProfileHMMTransitionPrior(),
                frozen_kernels={},
                dtype=tf.float32,
                **kwargs):
        super(ProfileHMMTransitioner, self).__init__(name="ProfileHMMTransitioner", dtype=dtype, **kwargs)
        self.transition_init = transition_init
        self.flank_init = flank_init
        self.prior = prior
        self.frozen_kernels = frozen_kernels
        self.epsilon = tf.constant(1e-32, dtype)
        self.approx_log_zero = tf.math.log(self.epsilon)
        
    def cell_init(self, cell):
        """ Automatically called when the owner cell is created.
        """
        self.length = cell.length
        self.num_states = cell.num_states
        self.num_states_implicit = cell.num_states_implicit
        #sub-arrays of the complete transition kernel for convenience
        #describes name and length for kernels of the categorical transition distributions
        #for states or families of states
        self.explicit_transition_kernel_parts = _make_explicit_transition_kernel_parts(self.length)
        self.implicit_transition_parts = _make_implicit_transition_parts(self.length)
        self.sparse_transition_indices_implicit = _make_sparse_transition_indices_implicit(self.length)
        self.sparse_transition_indices_explicit = _make_sparse_transition_indices_explicit(self.length)
        _assert_transition_init_kernel(self.transition_init, self.explicit_transition_kernel_parts)
        self.prior.load(self.dtype)
        
    def build(self, input_shape=None):
        # The (sparse) kernel is subdivided in groups of transitions.
        # To avoid error-prone slicing of a long array into smaller parts,
        # we store the parts as a dictionary and concatenate them later in the correct order.
        # The kernel is closely related to the transition matrix in the implicit model with deletion states.
        self.transition_kernel = {}
        for part_name, length, init, frozen, shared_with in self._get_kernel_parts_init_list():
            if (shared_with is None or all(s not in self.transition_kernel for s in shared_with)):
                k = self.add_weight(shape=[length], 
                                    initializer = init,
                                    name="transition_kernel_"+part_name,
                                    trainable=not frozen,
                                    dtype=self.dtype)
            else:
                for s in shared_with:
                    if s in self.transition_kernel:
                        k = self.transition_kernel[s]
                        break
            self.transition_kernel[part_name] = k
        
        # closely related to the initial probability of the left flank state
        self.flank_init_kernel = self.add_weight(shape=[1],
                                         initializer=self.flank_init,
                                         name="init_logit",
                                         dtype=self.dtype)
        
    def recurrent_init(self):
        """ Automatically called before each recurrent run. Should be used for setups that
            are only required once per application of the recurrent layer.
        """
        self.A = self.make_A_sparse()
        
    def make_flank_init_prob(self):
        return tf.math.sigmoid(self.flank_init_kernel)
        
    def make_initial_distribution(self):
        """Constructs the initial state distribution which depends on the transition probabilities.
        Returns:
            A probability distribution of shape: (q)
        """
        #state order: LEFT_FLANK, MATCH x length, INSERT x length-1, UNANNOTATED_SEGMENT, RIGHT_FLANK, TERMINAL
        init_flank_prob = self.make_flank_init_prob()
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
    
    def make_transition_kernel(self):
        """Concatenates the kernels of all transition types (e.g. match-to-match) in a consistent order.
        Returns:
            The concatenated kernel vector.
        """
        return tf.concat([self.transition_kernel[part_name] 
                          for part_name,_ in self.explicit_transition_kernel_parts], axis=0)
              
    def make_probs(self):
        """Computes all transition probabilities from kernels. Applies a softmax to the kernel values of 
            all outgoing edges of a state.
        Returns:
            A dictionary that maps transition types to probabilies. 
        """
        indices_explicit_dict = self.sparse_transition_indices_explicit
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
        """Computes all logarithmic transition probabilities in the implicit model. 
        Returns:
            A dictionary that maps transition types to probabilies. 
        """
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
        indices_implicit_dict = self.sparse_transition_indices_implicit
        indices_implicit = np.concatenate([indices_implicit_dict[part_name] 
                                                for part_name,_ in self.implicit_transition_parts], axis=0)
        log_A_sparse = tf.sparse.SparseTensor(
                            indices=indices_implicit, 
                            values=values, 
                            dense_shape=[self.num_states]*2)
        log_A_sparse = tf.sparse.reorder(log_A_sparse)
        return log_A_sparse
    
    def make_log_A(self):
        log_A = self.make_log_A_sparse()
        log_A = tf.sparse.to_dense(log_A, default_value=self.approx_log_zero)
        return log_A
    
    def make_A_sparse(self):
        log_A_sparse = self.make_log_A_sparse()
        A_sparse = tf.sparse.SparseTensor(
                            indices=log_A_sparse.indices, 
                            values=tf.math.exp(log_A_sparse.values), 
                            dense_shape=log_A_sparse.dense_shape)
        return A_sparse
        
    def make_A(self):
        A = self.make_A_sparse()
        A = tf.sparse.to_dense(A)
        return A
        
    def call(self, inputs):
        return tf.sparse.sparse_dense_matmul(inputs, self.A) 
    
    def get_prior_log_densities(self):
        return self.prior(self.make_probs(), self.make_flank_init_prob())
    
    def duplicate(self):
        return ProfileHMMTransitioner(
                transition_init = self.transition_init,
                flank_init = self.flank_init,
                prior = self.prior,
                frozen_kernels = self.frozen_kernels,
                dtype = self.dtype) 
    
    def _get_kernel_parts_init_list(self):
        """ Returns a list that specifies initialization data to the cell for all transition kernels.
            The list should contain 5-tuples: 
            (part_name : str, length : int, init : tf.initializer, frozen : bool, shared_with : list or None)
        """
        shared_kernels = [ ["unannotated_segment_loop", "right_flank_loop", "left_flank_loop"], 
                           ["unannotated_segment_exit", "right_flank_exit", "left_flank_exit"] ]
        #map each name to the list it is contained in
        #assume that shared_kernels contains each name at most once
        shared_kernel_dict = {} 
        for shared in shared_kernels: 
            for name in shared:
                shared_kernel_dict[name] = shared
        return [(part_name, 
                 length, 
                 self.transition_init[part_name], 
                 self.frozen_kernels.get(part_name, False), 
                 shared_kernel_dict.get(part_name, None)) 
                    for part_name, length in self.explicit_transition_kernel_parts]
    
    def __repr__(self):
        return f"ProfileHMMTransitioner(transition_init={self.transition_init}, flank_init={self.flank_init}, prior={self.prior}, frozen_kernels={self.frozen_kernels})"
    

def _make_explicit_transition_kernel_parts(length): 
    return [("begin_to_match", length), 
             ("match_to_end", length),
             ("match_to_match", length-1), 
             ("match_to_insert", length-1),
             ("insert_to_match", length-1), 
             ("insert_to_insert", length-1),
            #consider begin and end states as additional match states:
             ("match_to_delete", length), 
             ("delete_to_match", length),
             ("delete_to_delete", length-1),
             ("left_flank_loop", 1), 
             ("left_flank_exit", 1),
             ("unannotated_segment_loop", 1), 
             ("unannotated_segment_exit", 1),
             ("right_flank_loop", 1), 
             ("right_flank_exit", 1),
             ("end_to_unannotated_segment", 1), 
             ("end_to_right_flank", 1), 
             ("end_to_terminal", 1)]


def _make_implicit_transition_parts(length):
    return ([("left_flank_loop", 1),
               ("left_flank_to_match", length),
               ("left_flank_to_right_flank", 1),
               ("left_flank_to_unannotated_segment", 1),
               ("left_flank_to_terminal", 1),
               ("match_to_match", length-1),
               ("match_skip", int((length-1) * (length-2) / 2)),
               ("match_to_unannotated", length),
               ("match_to_right_flank", length),
               ("match_to_terminal", length),
               ("match_to_insert", length-1),
               ("insert_to_match", length-1),
               ("insert_to_insert", length-1),
               ("unannotated_segment_to_match", length),
               ("unannotated_segment_loop", 1),
               ("unannotated_segment_to_right_flank", 1),
               ("unannotated_segment_to_terminal", 1),
               ("right_flank_loop", 1),
               ("right_flank_exit", 1),
               ("terminal_self_loop", 1)])



def _make_sparse_transition_indices_implicit(length):
    """ Returns 2D indices for the kernel of a sparse (2L+3 x 2L+3) transition matrix without silent states.
        Assumes the following ordering of states: 
        LEFT_FLANK, MATCH x length, INSERT x length-1, UNANNOTATED_SEGMENT, RIGHT_FLANK, TERMINAL
    """
    a = np.arange(length+1)
    left_flank = 0
    first_insert = length + 1
    unanno_segment = 2*length
    right_flank = 2*length + 1
    terminal = 2*length + 2
    indices_dict = {
        "left_flank_loop" : [[left_flank, left_flank]],
        "left_flank_to_match" : np.stack([np.zeros(length)+left_flank, a[1:]], axis=1),
        "left_flank_to_right_flank" : [[left_flank, right_flank]],
        "left_flank_to_unannotated_segment" : [[left_flank, unanno_segment]],
        "left_flank_to_terminal" : [[left_flank, terminal]],
        "match_to_match" : np.stack([a[1:-1], a[1:-1]+1], axis=1),
        "match_skip" : np.concatenate([np.stack([np.zeros(length-i-1)+i, 
                                     np.arange(i+2, length+1)], axis=1)
            for i in range(1, length-1)
                ], axis=0),
        "match_to_unannotated" : np.stack([a[1:], np.zeros(length)+unanno_segment], axis=1),
        "match_to_right_flank" : np.stack([a[1:], np.zeros(length)+right_flank], axis=1),
        "match_to_terminal" : np.stack([a[1:], np.zeros(length)+terminal], axis=1),
        "match_to_insert" : np.stack([a[1:-1], a[:-2]+first_insert], axis=1),
        "insert_to_match" : np.stack([a[:-2]+first_insert, a[2:]], axis=1),
        "insert_to_insert" : np.stack([a[:-2]+first_insert]*2, axis=1),
        "unannotated_segment_to_match" : np.stack([np.zeros(length)+unanno_segment, a[1:]], axis=1),
        "unannotated_segment_loop" : [[unanno_segment, unanno_segment]],
        "unannotated_segment_to_right_flank" : [[unanno_segment, right_flank]],
        "unannotated_segment_to_terminal" : [[unanno_segment, terminal]],
        "right_flank_loop" : [[right_flank, right_flank]],
        "right_flank_exit" : [[right_flank, terminal]],
        "terminal_self_loop" : [[terminal, terminal]]}
    return indices_dict

def _make_sparse_transition_indices_explicit(length):
    """ Returns 2D indices for the (linear) kernel of a sparse (3L+3 x 3L+3) transition matrix with silent states.
        Assumes the following ordering of states:
        LEFT_FLANK, MATCH x length, INSERT x length-1, UNANNOTATED_SEGMENT, 
        RIGHT_FLANK, TERMINAL, BEGIN, END, DELETE x length
    """
    a = np.arange(length+1)
    left_flank = 0
    first_insert = length + 1
    unanno_segment = 2*length
    right_flank = 2*length + 1
    terminal = 2*length + 2
    begin = 2*length + 3
    end = 2*length + 4
    first_delete = 2*length + 5
    indices_dict = {
        "begin_to_match" : np.stack([np.zeros(length)+begin, a[1:]], axis=1),
        "match_to_end" : np.stack([a[1:], np.zeros(length)+end], axis=1),
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
            
def _assert_transition_init_kernel(kernel_init, parts):
    for part_name,_ in parts:
        assert part_name in kernel_init, "No initializer found for kernel " + part_name + "."
    for part_name in kernel_init.keys():
        assert part_name in [part[0] for part in parts], part_name + " is in the kernel init dict but there is no kernel part matching it. Wrong spelling?"
        
        
def _logsumexp(x, y):
    return tf.math.log(tf.math.exp(x) +  tf.math.exp(y))