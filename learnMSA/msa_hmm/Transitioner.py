import tensorflow as tf
import numpy as np
import learnMSA.msa_hmm.Initializers as initializers
import learnMSA.msa_hmm.Priors as priors
import learnMSA.msa_hmm.Configuration as config
from packaging import version
if version.parse(tf.__version__) < version.parse("2.11.0"):
    from tensorflow.python.training.tracking.data_structures import NoDependency #see https://github.com/tensorflow/tensorflow/issues/36916
else:
    from tensorflow.python.trackable.data_structures import NoDependency 

class ProfileHMMTransitioner(tf.keras.layers.Layer):
    """ A transitioner defines which transitions between HMM states are allowed, how they are initialized
        and how the transition matrix is represented (dense, sparse, other).
        The transitioner also holds a prior on the transition distributions. 
        This transitioner implements the default profile HMM logic with the additional Plan7 states.
    Args:
        transition_init: A list of dictionaries with initializers for each edge type, one per model. 
        flank_init: A list of dictionaries with initializers for the initial probability of the left flank state. 
        prior: A compatible prior that regularizes each transition type.
        frozer_kernels: A dictionary that can be used to omit parameter updates for certain kernels 
                        by adding "kernel_id" : False
    """
    def __init__(self, 
                transition_init = initializers.make_default_transition_init(),
                flank_init = initializers.make_default_flank_init(),
                prior = priors.ProfileHMMTransitionPrior(),
                frozen_kernels={},
                **kwargs):
        super(ProfileHMMTransitioner, self).__init__(**kwargs)
        transition_init = [transition_init] if isinstance(transition_init, dict) else transition_init 
        self.transition_init = NoDependency(transition_init)
        self.flank_init = [flank_init] if not hasattr(flank_init, '__iter__') else flank_init 
        self.prior = prior
        self.frozen_kernels = frozen_kernels
        self.epsilon = tf.constant(1e-32, self.dtype)
        self.approx_log_zero = tf.math.log(self.epsilon)
        
    def cell_init(self, cell):
        """ Automatically called when the owner cell is created.
        """
        self.length = cell.length
        assert len(self.length) == len(self.transition_init), \
            f"The number of transition initializers ({len(self.transition_init)}) should match the number of models ({len(self.length)})."
        assert len(self.length) == len(self.flank_init), \
            f"The number of flank initializers ({len(self.flank_init)}) should match the number of models ({len(self.length)})."
        self.num_states = cell.num_states
        self.num_states_implicit = cell.num_states_implicit
        self.max_num_states = cell.max_num_states
        self.num_models = cell.num_models
        #sub-arrays of the complete transition kernel for convenience
        #describes name and length for kernels of the categorical transition distributions
        #for states or families of states
        self.explicit_transition_kernel_parts = [_make_explicit_transition_kernel_parts(length) for length in self.length]
        self.implicit_transition_parts = [_make_implicit_transition_parts(length) for length in self.length]
        self.sparse_transition_indices_implicit = [_make_sparse_transition_indices_implicit(length) for length in self.length]
        self.sparse_transition_indices_explicit = [_make_sparse_transition_indices_explicit(length) for length in self.length]
        for init, parts in zip(self.transition_init, self.explicit_transition_kernel_parts):
            _assert_transition_init_kernel(init, parts)
        self.prior.load(self.dtype)
        
    def build(self, input_shape=None):
        # The (sparse) kernel is subdivided in groups of transitions.
        # To avoid error-prone slicing of a long array into smaller parts,
        # we store the parts as a dictionary and concatenate them later in the correct order.
        # The kernel is closely related to the transition matrix in the implicit model with deletion states.
        self.transition_kernel = []
        for model_kernel_parts in self._get_kernel_parts_init_list():
            model_transition_kernel = {}
            for i, (part_name, length, init, frozen, shared_with) in enumerate(model_kernel_parts):
                if (shared_with is None 
                    or all(s not in model_transition_kernel for s in shared_with)):
                    k = self.add_weight(shape=[length], 
                                        initializer = init,
                                        name="transition_kernel_"+part_name+"_"+str(i),
                                        trainable=not frozen,
                                        dtype=self.dtype)
                else:
                    for s in shared_with:
                        if s in model_transition_kernel:
                            k = model_transition_kernel[s]
                            break
                model_transition_kernel[part_name] = k
            self.transition_kernel.append(model_transition_kernel)
        
        # closely related to the initial probability of the left flank state
        self.flank_init_kernel = [self.add_weight(shape=[1],
                                         initializer=init,
                                         name="init_logit_"+str(i),
                                         dtype=self.dtype)
                                      for i,init in enumerate(self.flank_init)]
        self.built = True
        
    def recurrent_init(self):
        """ Automatically called before each recurrent run. Should be used for setups that
            are only required once per application of the recurrent layer.
        """
        self.A_sparse, self.implicit_log_probs, self.log_probs, self.probs = self.make_A_sparse(return_probs = True)
        self.A = tf.sparse.to_dense(self.A_sparse)
        
    def make_flank_init_prob(self):
        return tf.math.sigmoid(tf.stack(self.flank_init_kernel))
        
    def make_initial_distribution(self):
        """Constructs the initial state distribution per model which depends on the transition probabilities.
        Returns:
            A probability distribution per model. Shape: (1, k, q)
        """
        #state order: LEFT_FLANK, MATCH x length, INSERT x length-1, UNANNOTATED_SEGMENT, RIGHT_FLANK, TERMINAL
        init_flank_probs = self.make_flank_init_prob()
        log_init_flank_probs = tf.math.log(init_flank_probs)
        log_complement_init_flank_probs = tf.math.log(1-init_flank_probs)
        log_init_dists = []
        for i in range(self.num_models):
            log_init_match = (self.implicit_log_probs[i]["left_flank_to_match"] 
                          + log_complement_init_flank_probs[i]
                          - self.log_probs[i]["left_flank_exit"])
            log_init_right_flank = (self.implicit_log_probs[i]["left_flank_to_right_flank"] 
                                + log_complement_init_flank_probs[i] 
                                - self.log_probs[i]["left_flank_exit"])
            log_init_unannotated_segment = (self.implicit_log_probs[i]["left_flank_to_unannotated_segment"] 
                                        + log_complement_init_flank_probs[i] 
                                        - self.log_probs[i]["left_flank_exit"])
            log_init_terminal = (self.implicit_log_probs[i]["left_flank_to_terminal"] 
                             + log_complement_init_flank_probs[i] 
                             - self.log_probs[i]["left_flank_exit"] )
            log_init_insert = tf.zeros((self.length[i]-1), dtype=self.dtype) + self.approx_log_zero
            log_init_dist = tf.concat([log_init_flank_probs[i], 
                                        log_init_match, 
                                        log_init_insert, 
                                        log_init_unannotated_segment, 
                                        log_init_right_flank, 
                                        log_init_terminal], axis=0)
            log_init_dist = tf.pad(log_init_dist, 
                                   [[0, self.max_num_states - self.num_states[i]]], 
                                   constant_values = self.approx_log_zero)
            log_init_dists.append(log_init_dist)
        log_init_dists = tf.stack(log_init_dists, axis=0)
        log_init_dists = tf.expand_dims(log_init_dists, 0)
        init_dists = tf.math.exp(log_init_dists)
        return init_dists
    
    def make_transition_kernel(self):
        """Concatenates the kernels of all transition types (e.g. match-to-match) in a consistent order.
        Returns:
            The concatenated kernel vector.
        """
        concat_transition_kernels = []
        for part_names, kernel in zip(self.explicit_transition_kernel_parts, self.transition_kernel):
            concat_kernel = tf.concat([kernel[part_name] for part_name,_ in part_names], axis=0)
            concat_transition_kernels.append( concat_kernel )
        return concat_transition_kernels
              
    def make_probs(self):
        """Computes all transition probabilities from kernels. Applies a softmax to the kernel values of 
            all outgoing edges of a state.
        Returns:
            A dictionary that maps transition types to probabilies. 
        """
        model_prob_dicts = []
        for indices_explicit, parts, num_states, kernel in zip(self.sparse_transition_indices_explicit,
                                                                self.explicit_transition_kernel_parts,
                                                                self.num_states_implicit,
                                                                self.make_transition_kernel()):
            probs_dict = {}
            indices_explicit = np.concatenate([indices_explicit[part_name] 
                                                    for part_name,_ in parts], axis=0)
            # tf.sparse requires a strict row-major ordering of the indices
            # however, a custom ordering is more convenient in code
            # these indices revert a tf.sparse.reorder:
            a = np.argsort([i*num_states+j for i,j in indices_explicit])
            reverse_reorder_indices = np.argsort(a)
            sparse_kernel =  tf.sparse.SparseTensor(
                                        indices=indices_explicit, 
                                        values=kernel, 
                                        dense_shape=[num_states]*2)
            sparse_kernel = tf.sparse.reorder(sparse_kernel)
            probs = tf.sparse.softmax(sparse_kernel, name="A") #ignores implicit zeros
            probs_vec = tf.gather(probs.values, reverse_reorder_indices) #revert tf.sparse.reorder
            lsum = 0
            for part_name, length in parts:
                probs_dict[part_name] = probs_vec[lsum : lsum+length]
                lsum += length
            model_prob_dicts.append(probs_dict)
        return model_prob_dicts
    
    def make_log_probs(self):
        probs = self.make_probs()
        log_probs = [{key : tf.math.log(p) for key,p in model_probs.items()} for model_probs in probs]
        return log_probs, probs
    
    def make_implicit_log_probs(self):
        """Computes all logarithmic transition probabilities in the implicit model. 
        Returns:
            A dictionary that maps transition types to probabilies. 
        """
        log_probs, probs = self.make_log_probs()
        implicit_log_probs = []
        for p, length in zip(log_probs, self.length):
            #compute match_skip(i,j) = P(Mj+2 | Mi)  , L x L
            #considers "begin" as M0 and "end" as ML
            MD = tf.expand_dims(p["match_to_delete"], -1)
            DD = tf.concat([[0], p["delete_to_delete"]], axis=0)
            DD_cumsum = tf.math.cumsum(DD)
            DD = tf.expand_dims(DD_cumsum, 0) - tf.expand_dims(DD_cumsum, 1)
            DM = tf.expand_dims(p["delete_to_match"], 0)
            M_skip = MD + DD + DM 
            upper_triangle = tf.linalg.band_part(tf.ones([length-2]*2, dtype=self.dtype), 0, -1)
            entry_add = _logsumexp(p["begin_to_match"], 
                                   tf.concat([[self.approx_log_zero], M_skip[0, :-1]], axis=0))
            exit_add = _logsumexp(p["match_to_end"], 
                                  tf.concat([M_skip[1:,-1], [self.approx_log_zero]], axis=0))
            imp_probs = {}
            imp_probs["match_to_match"] = p["match_to_match"]
            imp_probs["match_to_insert"] = p["match_to_insert"]
            imp_probs["insert_to_match"] = p["insert_to_match"]
            imp_probs["insert_to_insert"] = p["insert_to_insert"]
            imp_probs["left_flank_loop"] = p["left_flank_loop"]
            imp_probs["right_flank_loop"] = p["right_flank_loop"]
            imp_probs["right_flank_exit"] = p["right_flank_exit"]
            imp_probs["match_skip"] = tf.boolean_mask(M_skip[1:-1, 1:-1], 
                                     mask=tf.cast(upper_triangle, dtype=tf.bool)) 
            imp_probs["left_flank_to_match"] = p["left_flank_exit"] + entry_add
            imp_probs["left_flank_to_right_flank"] = (p["left_flank_exit"] + M_skip[0, -1] 
                                                      + p["end_to_right_flank"])
            imp_probs["left_flank_to_unannotated_segment"] = (p["left_flank_exit"] + M_skip[0, -1] 
                                                              + p["end_to_unannotated_segment"])
            imp_probs["left_flank_to_terminal"] = (p["left_flank_exit"] + M_skip[0, -1] 
                                                   + p["end_to_terminal"])
            imp_probs["match_to_unannotated"] = exit_add + p["end_to_unannotated_segment"]
            imp_probs["match_to_right_flank"] = exit_add + p["end_to_right_flank"]
            imp_probs["match_to_terminal"] = exit_add + p["end_to_terminal"]
            imp_probs["unannotated_segment_to_match"] = p["unannotated_segment_exit"] + entry_add
            imp_probs["unannotated_segment_loop"] = _logsumexp(p["unannotated_segment_loop"], 
                                                               (p["unannotated_segment_exit"] 
                                                                    + M_skip[0, -1] 
                                                                    + p["end_to_unannotated_segment"]))
            imp_probs["unannotated_segment_to_right_flank"] = (p["unannotated_segment_exit"] 
                                                               + M_skip[0, -1] 
                                                               + p["end_to_right_flank"])
            imp_probs["unannotated_segment_to_terminal"] = (p["unannotated_segment_exit"] 
                                                            + M_skip[0, -1] 
                                                            + p["end_to_terminal"])
            imp_probs["terminal_self_loop"] = tf.zeros((1), dtype=self.dtype)
            implicit_log_probs.append(imp_probs)
        return implicit_log_probs, log_probs, probs
    
    def make_log_A_sparse(self, return_probs=False):
        """
        Returns:
            A 3D sparse tensor of dense shape (k, q, q) representing 
            the logarithmic transition matricies for k models.
        """
        implicit_log_probs, log_probs, probs = self.make_implicit_log_probs()
        values_all_models, indices_all_models = [], []
        for i, (p, parts, indices) in enumerate(zip(implicit_log_probs, 
                                                           self.implicit_transition_parts, 
                                                           self.sparse_transition_indices_implicit)):
            values = tf.concat([p[part_name] for part_name,_ in parts], axis=0)
            indices_concat = np.concatenate([indices[part_name] for part_name,_ in parts], axis=0)
            indices_concat = np.pad(indices_concat, ((0,0), (1,0)), constant_values=i)
            values_all_models.append(values)
            indices_all_models.append(indices_concat)
        values_all_models = tf.concat(values_all_models, axis=0) #"model major" order
        indices_all_models = np.concatenate(indices_all_models, axis=0)
        log_A_sparse = tf.sparse.SparseTensor(
                            indices=indices_all_models, 
                            values=values_all_models, 
                            dense_shape=[self.num_models] + [self.max_num_states]*2)
        log_A_sparse = tf.sparse.reorder(log_A_sparse)
        if return_probs:
            return log_A_sparse, implicit_log_probs, log_probs, probs
        else:
            return log_A_sparse
    
    def make_log_A(self):
        """
        Returns:
            A 3D dense tensor of shape (k, q, q) representing 
            the logarithmic transition matricies for k models.
        """
        log_A = self.make_log_A_sparse()
        log_A = tf.sparse.to_dense(log_A, default_value=self.approx_log_zero)
        return log_A
    
    def make_A_sparse(self, return_probs=False):
        """
        Returns:
            A 3D sparse tensor of dense shape (k, q, q) representing 
            the transition matricies for k models.
        """
        if return_probs:
            log_A_sparse, *p = self.make_log_A_sparse(True)
        else:
            log_A_sparse = self.make_log_A_sparse(False)
        A_sparse = tf.sparse.SparseTensor(
                            indices=log_A_sparse.indices, 
                            values=tf.math.exp(log_A_sparse.values), 
                            dense_shape=log_A_sparse.dense_shape)
        if return_probs:
            return A_sparse, *p
        else:
            return A_sparse
        
    def make_A(self):
        """
        Returns:
            A 3D dense tensor of shape (k, q, q) representing 
            the transition matricies for k models.
        """
        A = self.make_A_sparse()
        A = tf.sparse.to_dense(A)
        return A
        
    def call(self, inputs):
        """ 
        Args: 
                inputs: Shape (k, b, q)
        Returns:
                Shape (k, b, q)
        """
        #batch matmul of k inputs with k matricies
        return tf.matmul(inputs, self.A)
    
    def get_prior_log_densities(self):
        return self.prior(self.make_probs(), self.make_flank_init_prob())
    
    def duplicate(self, model_indices=None):
        if model_indices is None:
            model_indices = range(len(self.transition_init))
        sub_transition_init = []
        sub_flank_init = []
        for i in model_indices:
            transition_init_dict = {key : tf.constant_initializer(kernel.numpy())
                                       for key, kernel in self.transition_kernel[i].items()}
            sub_transition_init.append(transition_init_dict)
            sub_flank_init.append(tf.constant_initializer(self.flank_init_kernel[i].numpy()))
        transitioner_copy = ProfileHMMTransitioner(
                                        transition_init = sub_transition_init,
                                        flank_init = sub_flank_init,
                                        prior = self.prior,
                                        frozen_kernels = self.frozen_kernels,
                                        dtype = self.dtype) 
        return transitioner_copy
    
    #configure the Transitioner for the backward recursion
    def transpose(self):
        self.A = tf.transpose(self.A, (0,2,1))
        
    
    def _get_kernel_parts_init_list(self):
        """ Returns a list of lists that specifies initialization data to the cell for all transition kernels.
            The outer list contains one list per model. The inner list contains 5-tuples: 
            (part_name : str, length : int, init : tf.initializer, frozen : bool, shared_with : list or None)
        """
        #assume that shared_kernels contains each name at most once
        shared_kernels = [ ["right_flank_loop", "left_flank_loop"], 
                           ["right_flank_exit", "left_flank_exit"] ]
        #map each name to the list it is contained in
        shared_kernel_dict = {} 
        for shared in shared_kernels: 
            for name in shared:
                shared_kernel_dict[name] = shared
        kernel_part_list = []
        for init, parts in zip(self.transition_init, self.explicit_transition_kernel_parts):
            kernel_part_list.append( [(part_name, 
                                     length, 
                                     init[part_name], 
                                     self.frozen_kernels.get(part_name, False), 
                                     shared_kernel_dict.get(part_name, None)) 
                                        for part_name, length in parts] )
        return kernel_part_list
        
    def _pad_and_stack(self, dicts):
        # takes a list of dictionaries with the same keys that map to arrays of different lengths
        # returns a dictionary where each key is mapped to a stacked array with zero padding
        if len(dicts) == 1:
            return dicts[0]
        transposed = {}
        for d in dicts:
            for k,a in d.items():
                transposed.setdefault(k, []).append(a)
        padded_and_stacked = {k : tf.keras.preprocessing.sequence.pad_sequences(arrays, 
                                                                                dtype=arrays[0].dtype.name, 
                                                                                padding="post",
                                                                                value=self.approx_log_zero) 
                              for k,arrays in transposed.items()}
        return padded_and_stacked
    
    def get_config(self):
        config = super(ProfileHMMTransitioner, self).get_config()
        for key in self.transition_kernel[0].keys():
            config[key] = [self.transition_kernel[i][key].numpy() for i in range(self.num_models)]
        config.update({
            "num_models" : len(self.transition_kernel),
            "flank_init" : [k.numpy() for k in self.flank_init_kernel],
            "prior" : self.prior,
            "frozen_kernels" : self.frozen_kernels
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        transition_init = [{} for i in range(config.pop("num_models"))]
        for key,_ in _make_explicit_transition_kernel_parts(1):
            kernels = config.pop(key)
            for i,d in enumerate(transition_init):
                d[key] = initializers.ConstantInitializer(kernels[i])
        config["transition_init"] = transition_init
        config["flank_init"] = [initializers.ConstantInitializer(k) for k in config["flank_init"]]
        return cls(**config)
    
    def __repr__(self):
        return f"ProfileHMMTransitioner(\n transition_init={config.as_str(self.transition_init[0], 2, '    ', ' , ')},\n flank_init={self.flank_init[0]},\n prior={self.prior},\n frozen_kernels={self.frozen_kernels})"
    

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
    a = np.arange(length+1, dtype=np.int64)
    left_flank = 0
    first_insert = length + 1
    unanno_segment = 2*length
    right_flank = 2*length + 1
    terminal = 2*length + 2
    zeros = np.zeros(length, dtype=a.dtype)
    indices_dict = {
        "left_flank_loop" : [[left_flank, left_flank]],
        "left_flank_to_match" : np.stack([zeros+left_flank, a[1:]], axis=1),
        "left_flank_to_right_flank" : [[left_flank, right_flank]],
        "left_flank_to_unannotated_segment" : [[left_flank, unanno_segment]],
        "left_flank_to_terminal" : [[left_flank, terminal]],
        "match_to_match" : np.stack([a[1:-1], a[1:-1]+1], axis=1),
        "match_skip" : np.concatenate([np.stack([zeros[:-i-1]+i, 
                                     np.arange(i+2, length+1)], axis=1)
            for i in range(1, length-1)
                ], axis=0),
        "match_to_unannotated" : np.stack([a[1:], zeros+unanno_segment], axis=1),
        "match_to_right_flank" : np.stack([a[1:], zeros+right_flank], axis=1),
        "match_to_terminal" : np.stack([a[1:], zeros+terminal], axis=1),
        "match_to_insert" : np.stack([a[1:-1], a[:-2]+first_insert], axis=1),
        "insert_to_match" : np.stack([a[:-2]+first_insert, a[2:]], axis=1),
        "insert_to_insert" : np.stack([a[:-2]+first_insert]*2, axis=1),
        "unannotated_segment_to_match" : np.stack([zeros+unanno_segment, a[1:]], axis=1),
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
    a = np.arange(length+1, dtype=np.int64)
    left_flank = 0
    first_insert = length + 1
    unanno_segment = 2*length
    right_flank = 2*length + 1
    terminal = 2*length + 2
    begin = 2*length + 3
    end = 2*length + 4
    first_delete = 2*length + 5
    zeros = np.zeros(length, dtype=a.dtype)
    indices_dict = {
        "begin_to_match" : np.stack([zeros+begin, a[1:]], axis=1),
        "match_to_end" : np.stack([a[1:], zeros+end], axis=1),
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