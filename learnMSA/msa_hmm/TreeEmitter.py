from learnMSA.msa_hmm.Emitter import ProfileHMMEmitter
from learnMSA.msa_hmm.Utility import get_num_states
import learnMSA.msa_hmm.Initializers as initializers
import numpy as np
import sys
import tensorflow as tf
import copy

# tmp include
sys.path.insert(0, "../TensorTree")
import tensortree 

tensortree.set_backend("tensorflow")


r""" A special emitter when the observations are related by a tree.
    The tree contains each input sequence as a leaf and assumes that there are no other leaves.
    The ancestral tree is the remaining tree when pruning all leaves. Its nodes are the ancestral nodes.

    Current assumptions/limitations:
    1 Only full leaf sets are supported (no backbone support).
    2 Sequences are only allowed to connect to the leaves of the ancestral tree ("cluster trees"). 
    3 The topology is fixed during the whole alignment process. 

    E.g.

    Fasta:                  Tree:
    >A                         R
    ..                        / \
    >B                       U   V
    ..                      / \ / \
    >C                      A B C  D
    ..
    >D
    ..

    Then the tree U - R - V is the ancestral tree.
    
    The tree emitter models amino acid distributions at the leaves of the ancestral tree.
    The emission probabilities are computed based on which ancestral leaf each sequence is connected to.

    Auxiliary loss:
    The tree emitter adds an auxiliary loss that maximizes the likelihood of the ancestral tree.
"""
class TreeEmitter(ProfileHMMEmitter): 

    def __init__(self, 
                 tree_handler : tensortree.TreeHandler,
                 emission_init = initializers.make_default_emission_init(),
                 insertion_init = initializers.make_default_insertion_init(),
                 prior = None,
                 tree_loss_weight = 0.1):
        super(TreeEmitter, self).__init__(emission_init, insertion_init, 
                                          prior, frozen_insertions=True)
        self.tree_handler = tree_handler
        self.ancestral_tree_handler = copy.deepcopy(tree_handler)
        self.ancestral_tree_handler.prune()
        self.ancestral_tree_handler.update()
        # the tree handler defines both the mapping of the sequences (leaves) 
        # to the ancestral nodes and the ancestral topology
        
        # relies on assumption 2
        # todo: make this more general; allow connections to internal ancestral nodes
        self.cluster_indices = tree_handler.get_parent_indices_by_height(0)
        self.cluster_indices -= tree_handler.num_leaves # indices are sorted by height
        self.num_clusters = np.unique(self.cluster_indices).size
        self.tree_loss_weight = tree_loss_weight
        self.rate_matrix, self.equilibrium = _make_default_rate_matrix()


    def set_lengths(self, lengths):
        super(TreeEmitter, self).set_lengths(lengths)
        branch_lengths = np.concatenate([self.ancestral_tree_handler.branch_lengths]*self.num_models, axis=1)
        self.ancestral_tree_handler.set_branch_lengths(branch_lengths)

    
    def build(self, input_shape):
        if self.built:
            return
        s = input_shape[-1]-1 # substract one for terminal symbol
        self.emission_kernel = [self.add_weight(
                                    shape=(self.num_clusters, length, s), 
                                    initializer=init, 
                                    name="emission_kernel_"+str(i)) 
                                for i,(length, init) in enumerate(zip(self.lengths, self.emission_init))]
        self.insertion_kernel = [self.add_weight(
                                    shape=(self.num_clusters, s),
                                    initializer=init,
                                    name="insertion_kernel_"+str(i),
                                    trainable=not self.frozen_insertions) 
                                for i,init in enumerate(self.insertion_init)]
        if self.prior is not None:
            self.prior.build()
        self.built = True


    def recurrent_init(self):
        self.B = self.make_B()
        self.B_transposed = tf.transpose(self.B, [0,1,3,2])


    def call(self, inputs, indices, end_hints=None, training=False):
        """ 
        Args: 
                inputs: A tensor of shape (num_models, b, L , s) 
                indices: A tensor of shape (num_models, b) that contains the index of each input sequence.
        Returns:
                A tensor with emission probabilities of shape (num_models, b, L, q) where "..." is identical to inputs.
        """
        input_shape = tf.shape(inputs)
        B = self.B_transposed[..., :input_shape[-1],:]
        # we have to select the correct entry from the second dimension of B
        # for each input sequence
        cluster_indices = tf.gather(self.cluster_indices, indices)
        B = tf.gather(B, cluster_indices, batch_dims=1)
        return self._compute_emission_probs(inputs, B, input_shape, B_contains_batch=True)
    

    # computes the likelihood of the ancestral tree
    def get_aux_loss(self):

        # compute the likelihood of the ancestral tree with TensorTree
        leaves = self.B[..., 1:max(self.lengths)+1, :20] # only consider match positions and standard amino acids
        leaves = tf.transpose(leaves, [2,0,1,3])
        leaves /= tf.math.maximum(tf.reduce_sum(leaves, axis=-1, keepdims=True), 1e-16) #re-normalize

        anc_loglik = self._compute_anc_tree_loglik(leaves)

        loss = -self.tree_loss_weight * tf.reduce_mean(anc_loglik)

        return loss
    

    def _compute_anc_tree_loglik(self, leaf_probs):
        """ 
        Args:
            leaf_probs: A tensor of shape (num_leaves, num_models, model_length, 20) that contains the 
                        amino acid emission probabilities at the leaves of the ancestral tree.
        Returns: 
            loglik per model, averaged over model length
        """
        tree_handler = self.ancestral_tree_handler
        branch_lengths = self.ancestral_tree_handler.branch_lengths
        anc_loglik = tensortree.model.loglik(leaf_probs, tree_handler, self.rate_matrix, branch_lengths, tf.math.log(self.equilibrium))
        
        #mask out padding states and average over model length
        mask = tf.cast(tf.sequence_mask(self.lengths), anc_loglik.dtype)
        anc_loglik = tf.reduce_sum(anc_loglik * mask, axis=1) / tf.reduce_sum(mask, axis=1)
        
        return anc_loglik
    

    def get_prior_log_density(self):
        B_mod = tf.reshape(self.B, (self.num_models * self.num_clusters, self.num_states, self.emission_kernel.shape[-1]))
        priors = self.prior(B_mod, lengths=self.lengths)
        priors = tf.reshape(priors, (self.num_models, self.num_clusters, priors.shape[-1]))
        return tf.reduce_mean(priors, axis=1) # average over clusters
    

    def duplicate(self, model_indices=None, share_kernels=False):
        if model_indices is None:
            model_indices = range(len(self.emission_init))
        sub_emission_init = [initializers.ConstantInitializer(self.emission_kernel[i].numpy()) for i in model_indices]
        sub_insertion_init = [initializers.ConstantInitializer(self.insertion_kernel[i].numpy()) for i in model_indices]
        emitter_copy = TreeEmitter(
                             tree_handler = self.tree_handler,
                             emission_init = sub_emission_init,
                             insertion_init = sub_insertion_init,
                             prior = self.prior,
                             tree_loss_weight = self.tree_loss_weight) 
        if share_kernels:
            emitter_copy.emission_kernel = self.emission_kernel
            emitter_copy.insertion_kernel = self.insertion_kernel
            emitter_copy.built = True
        return emitter_copy
    

    def get_config(self):
        config = super(TreeEmitter, self).get_config()
        config.update({
        "tree_handler": self.tree_handler,
        "tree_loss_weight": self.tree_loss_weight
        })
        return config
    

    def __repr__(self):
        return f"TreeEmitter()"
    



    

def _make_default_rate_matrix(num_models=1):
    R, p = tensortree.substitution_models.LG()
    R = R[tf.newaxis]
    p = p[tf.newaxis]
    Q = tensortree.backend.make_rate_matrix(R, p)
    return Q, p