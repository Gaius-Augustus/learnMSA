from learnMSA.msa_hmm.Emitter import ProfileHMMEmitter
from learnMSA.msa_hmm.Utility import get_num_states, deserialize
import learnMSA.msa_hmm.Initializers as initializers
from learnMSA.msa_hmm.Utility import inverse_softplus, perturbate
import numpy as np
import sys
import tensorflow as tf
import tensortree 
from tensortree import model as tree_model

tensortree.set_backend("tensorflow")


class TreeEmitter(ProfileHMMEmitter): 
    r""" A special emitter when the observations are related by a tree.
    The tree is specified by a TreeHandler object.
    Each input sequence to be aligned must map bijectively to tree leaves.
    Otherwise the tree topology is not constrained.

    We'll denote the remaining tree when pruning all (sequence-)leaves
    from the input tree as the ancestral tree. Its nodes are the ancestral 
    nodes.

    Current assumptions/limitations:
    1 All input sequences must be already represented in the tree. There is no 
    automatic mapping mechanism to incorporate new sequences into the tree.
    2 Sequences are only allowed to connect to the leaves of the ancestral tree 
    ("cluster trees"). 
    3 The full ancestral tree will be processed by the tree emitter. There is 
    no reduction mechanism to handle large trees, so the ancestral topology 
    must be relatively small.
    4 The topology is fixed during the whole alignment process. 

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

    Then the tree U <- R -> V is the ancestral tree.
    
    The tree emitter models amino acid distributions at the leaves of the 
    ancestral tree explicitly and infers the rest using the tree topology, 
    branch lengths. The emission probabilities are computed based on which 
    ancestral leaf each sequence is connected to.

    Auxiliary loss:
    The tree emitter adds an auxiliary loss that maximizes the likelihood of 
    the ancestral tree.
    """

    def __init__(self, 
                 tree_handler : tensortree.TreeHandler,
                 emission_init = initializers.make_default_emission_init(),
                 insertion_init = initializers.make_default_insertion_init(),
                 branch_lengths_init = None,
                 prior = None,
                 tree_loss_weight = 1.0,
                 perturbate_cluster_index_prob=0.0,
                 propagate_root=True,
                 seperate_equilibrium=True):
        """
        Args:
            tree_handler: A TreeHandler object that defines the full tree 
                topology with the sequences as leaves.
            emission_init: Initializer for the emission kernel. Should be a 
                list when
            insertion_init: A list of initializers for the insertion kernels.
            prior: A prior object that defines a prior distribution over the 
                emission probabilities.
            tree_loss_weight: A scalar that weights the tree loss in the total 
                loss. 
            perturbate_cluster_index_prob: A scalar that defines the 
                probability of randomly perturbating the cluster index in each 
                step.
            propagate_root: If True, the emission parameters of the clusters
                will be inferred by propagating a root kernel down the tree.
            seperate_equilibrium:
                Whether to model the equilibrium distributions separately.
        """
        super(TreeEmitter, self).__init__(emission_init, insertion_init, 
                                          prior, frozen_insertions=True)
        
        # the tree handler defines both the mapping of the sequences (leaves) 
        # to the ancestral nodes and the ancestral topology
        self.tree_handler = tree_handler

        # prune all leaves from the input tree to get the ancestral tree
        self.ancestral_tree_handler = tensortree.TreeHandler.copy(tree_handler)
        self.ancestral_tree_handler.prune()
        self.ancestral_tree_handler.update()
        self.num_anc_nodes = self.ancestral_tree_handler.num_nodes

        if branch_lengths_init is None:
            # if no branch initializer is provided, infer from tree
            W = inverse_softplus(
                tree_handler.branch_lengths[tree_handler.num_leaves:]
            )
            branch_lengths_init = initializers.ConstantInitializer(W)
        self.branch_lengths_init = branch_lengths_init
        
        # relies on assumption 2
        # todo: make this more general; allow connections to internal ancestral 
        # nodes
        leaf_parent_indices = self.tree_handler.get_parent_indices_by_height(0)
        # make a copy to not affect the tree handler
        self.cluster_indices = leaf_parent_indices.copy()
        # convert to range 0..num_ancestral_nodes-1
        self.cluster_indices -= self.tree_handler.num_leaves 
        self.num_clusters = np.unique(self.cluster_indices).size
        self.cluster_sizes = np.bincount(self.cluster_indices).astype(np.float32)
        self.tree_loss_weight = tf.Variable(tree_loss_weight, trainable=False)

        if not propagate_root:
            self.rate_matrix, self.equilibrium = _make_default_rate_matrix()
        else:
            # otherwise these are created in build
            pass 

        # make this a variable and change it manually during training
        self.perturbate_cluster_index_prob = tf.Variable(
            perturbate_cluster_index_prob, trainable=False
        )
        self.propagate_root = propagate_root
        self.seperate_equilibrium = seperate_equilibrium
                                                         

    def set_lengths(self, lengths):
        super(TreeEmitter, self).set_lengths(lengths)
        branch_lengths = np.concatenate(
            [self.ancestral_tree_handler.branch_lengths]*self.num_models, 
            axis=1
        )
        self.ancestral_tree_handler.set_branch_lengths(
            branch_lengths, update_phylo_tree=False
        )

    
    def build(self, input_shape):
        if self.built:
            return
        s = input_shape[-1]-1 # substract one for terminal symbol

        if self.propagate_root:
            # model the cluster kernels implicitly
            self.emission_kernel = [
                self.add_weight(
                    shape=(length, s), 
                    initializer=init, 
                    name="emission_kernel_"+str(i)
                ) 
                for i,(length, init) in enumerate(
                    zip(self.lengths, self.emission_init)
                )
            ]
            self.insertion_kernel = [
                self.add_weight(
                    shape=(s,),
                    initializer=init,
                    name="insertion_kernel_"+str(i),
                    trainable=not self.frozen_insertions) 
                for i,init in enumerate(self.insertion_init)
            ]
        else:
            # model the cluster kernels explicitly
            self.emission_kernel = [
                self.add_weight(
                    shape=(self.num_clusters, length, s), 
                    initializer=init, 
                    name="emission_kernel_"+str(i)
                ) 
                for i,(length, init) in enumerate(
                    zip(self.lengths, self.emission_init)
                )
            ]
            self.insertion_kernel = [
                self.add_weight(
                    shape=(self.num_clusters, s),
                    initializer=init,
                    name="insertion_kernel_"+str(i),
                    trainable=not self.frozen_insertions) 
                for i,init in enumerate(self.insertion_init)
            ]

        self.branch_lengths_kernel = self.add_weight(
            shape=(self.num_anc_nodes-1, self.num_models),
            initializer=self.branch_lengths_init,
            name="branch_lengths_kernel"
        )

        if self.propagate_root:
            I = initializers.make_LG_init()
            self.exchangeability_init, self.equilibrium_init = I
            self.exchangeability_kernel = self.add_weight(
                shape=[
                    self.num_anc_nodes-1, 
                    self.num_models, 
                    max(self.lengths)+1, 
                    20, 
                    20
                ], 
                name="exchangeability_kernel", 
                initializer=self.exchangeability_init
            )
            if self.seperate_equilibrium:
                self.equilibrium_kernel = self.add_weight(
                    shape=[
                        self.num_models, 
                        20
                    ], 
                    name="equilibrium_kernel",
                    initializer=self.equilibrium_init,
                    trainable=True
                )

        if self.prior is not None:
            self.prior.build()
        self.built = True


    def make_B(self):
        B = super(TreeEmitter, self).make_B()
        if self.propagate_root:

            # transpose and get the amino acid distributions
            B_amino_root = B[..., :20] #(num_models, length, 20)
            length = tf.shape(B_amino_root)[1]
            B_amino_rest = B[..., 20:]

            # tensortree expects a special first dimension, the 
            # node-dimension, in this case, we start with a single root
            B_amino_root = B_amino_root[tf.newaxis,:,:max(self.lengths)+1]

            # setup the rate matrix and equilibrium distribution
            R = tensortree.backend.make_symmetric_pos_semidefinite(
                self.exchangeability_kernel
            )

            # the root/equilibrium distribution 
            if self.seperate_equilibrium:
                pi = tensortree.backend.make_equilibrium(
                    self.equilibrium_kernel[tf.newaxis,:,tf.newaxis]
                )
            else:
                pi = B_amino_root

            Q = tensortree.backend.make_rate_matrix(R, pi)

            L = self.get_branch_lengths()
            # broadcast along the model length dimension
            L = L[:,:,tf.newaxis]

            T = tensortree.backend.make_transition_probs(Q, L)
            
            B_amino = tree_model.propagate(
                B_amino_root, 
                self.ancestral_tree_handler, 
                T
            )

            # B_amino has incomplete model dimension, as it does not include all
            # insertion states yet, take the first insertion state and
            # replicate it for all others
            max_num_states = max(get_num_states(self.lengths))
            num_states_rest = max_num_states - (max(self.lengths) + 1) - 1
            rep_states = tf.repeat(
                B_amino[:,:,:1], #first insertion state
                num_states_rest,
                axis=2
            )
            terminal_state = B[tf.newaxis,:,-1:,:20]
            terminal_state = tf.repeat(
                terminal_state,
                rep_states.shape[0],
                axis=0
            )
            # concatenate the insertion states
            B_amino = tf.concat([B_amino, rep_states, terminal_state], axis=2)

            # get only the distributions of the clusters, not the root
            B_amino = B_amino[:self.num_clusters]
            # transpose to get the correct shape with models as first dimension
            B_amino = tf.transpose(B_amino, [1,0,2,3])

            # add back the rest of the distributions and renormalize
            B_amino_rest = tf.repeat(
                B_amino_rest[:,tf.newaxis], self.num_clusters, axis=1
            )
            B = tf.concat([B_amino, B_amino_rest], axis=-1)

            # renormalize
            B /= tf.math.maximum(
                tf.reduce_sum(B, axis=-1, keepdims=True), 1e-16
            )

        return B


    def recurrent_init(self, indices):

        self.indices = indices
        self.B = self.make_B()
        self.B_transposed = tf.transpose(self.B, [0,1,3,2])


    def call(self, inputs, end_hints=None, training=False):
        """ 
        Args: 
                inputs: A tensor of shape (num_models, b, L , s) 
        Returns:
                A tensor with emission probabilities of shape 
                (num_models, b, L, q) where "..." is identical to inputs.
        """
        input_shape = tf.shape(inputs)
        B = self.B_transposed[..., :input_shape[-1],:]
        # we have to select the correct parameters for each input sequence
        cluster_indices = tf.gather(self.cluster_indices, self.indices)
        if training:
            cluster_indices = perturbate(
                cluster_indices, 
                self.perturbate_cluster_index_prob, 
                self.num_clusters
            )
        B = tf.gather(B, cluster_indices, batch_dims=1)
        return self._compute_emission_probs(
            inputs, B, input_shape, B_contains_batch=True
        )
    

    def get_aux_loss(self, aggregate_models=True):
        if self.propagate_root:
            return tf.constant(0.0)
        else:
            # computes the tree loss
            # i.e. the likelihood of the ancestral tree with TensorTree
            # only consider match positions and standard amino acids
            leaves = self.B[..., 1:max(self.lengths)+1, :20] 
            leaves = tf.transpose(leaves, [1,0,2,3])
            #re-normalize
            leaves /= tf.math.maximum(
                tf.reduce_sum(leaves, axis=-1, keepdims=True), 1e-16
            ) 

            # weight by the number of sequences in each cluster
            leaves = tf.math.pow(
                leaves,
                self.cluster_sizes[:, tf.newaxis, tf.newaxis, tf.newaxis],
            )

            # compute the loglikelihood of the leaves given the ancestral tree
            anc_loglik = self.compute_anc_tree_loglik(leaves)

            # weight and average the loglikelihood over all models
            if aggregate_models:
                loss = -self.tree_loss_weight * tf.reduce_mean(anc_loglik)
            else:
                loss = -self.tree_loss_weight * anc_loglik

            return loss
    

    def compute_anc_tree_loglik(self, leaf_probs, average_over_length=False):
        """ 
        Args:
            leaf_probs: A tensor of shape (num_leaves, num_models, model_length, 
                20) that contains the amino acid emission probabilities at the 
                leaves of the ancestral tree.
        Returns: 
            loglik per model, summed or averaged over model length
        """
        tree_handler = self.ancestral_tree_handler
        branch_lengths = self.get_branch_lengths()
        transition_probs = tensortree.backend.make_transition_probs(
            self.rate_matrix, 
            branch_lengths
        )
        # add the length dimension
        transition_probs = transition_probs[:,:,tf.newaxis]
        anc_loglik = tree_model.loglik(
            leaf_probs, 
            tree_handler, 
            transition_probs,
            tf.math.log(self.equilibrium)
        )
        #mask out padding states and average over model length
        mask = tf.cast(tf.sequence_mask(self.lengths), anc_loglik.dtype)
        anc_loglik = tf.reduce_sum(anc_loglik * mask, axis=1) 
        if average_over_length:
            anc_loglik /= tf.reduce_sum(mask, axis=1)
        
        return anc_loglik
    

    def get_branch_lengths(self):
        return tensortree.backend.make_branch_lengths(self.branch_lengths_kernel)
    

    def get_prior_log_density(self):
        B_mod = tf.reshape(
            self.B, 
            (self.num_models * self.num_clusters, self.B.shape[-2], self.B.shape[-1])
        )
        priors = self.prior(
            B_mod, lengths=np.repeat(self.lengths, self.num_clusters).astype(np.int32)
        )
        priors = tf.reshape(
            priors, (self.num_models, self.num_clusters, priors.shape[-1])
        )
        return tf.reduce_mean(priors, axis=1) # average over clusters


    def make_B_amino(self):
        """ A variant of make_B used for plotting the HMM. Can be overridden 
        for more complex emissions. Per default this is equivalent to make_B
        """
        return self.make_B() # currently only the first cluster is used for plotting
    

    def duplicate(self, model_indices=None, share_kernels=False):
        if model_indices is None:
            model_indices = range(len(self.emission_init))
        sub_emission_init = [
            initializers.ConstantInitializer(self.emission_kernel[i].numpy()) 
            for i in model_indices
        ]
        sub_insertion_init = [
            initializers.ConstantInitializer(self.insertion_kernel[i].numpy()) 
            for i in model_indices
        ]

        # check if the tree handlers branch lengths are in their original shape,
        # i.e. (num_nodes-1, 1) as loaded from the newick file
        # if so, we need to repeat them appropriately for the multi-dimensional 
        # case
        if (len(self.tree_handler.branch_lengths.shape) == 2 
            and self.tree_handler.branch_lengths.shape[-1] == 1):
            # repeat the branch lengths for each model
            rep_branch_lengths = np.repeat(
                self.tree_handler.branch_lengths, 
                self.num_models, 
                axis=1
            )
            self.tree_handler.set_branch_lengths(
                rep_branch_lengths,
                update_phylo_tree=False
            )

        # keep the leaves but update the ancestral branch lengths
        # according to the kernels of this emitter
        num_leaves = self.tree_handler.num_leaves
        new_anc_branch_lengths = tensortree.backend.make_branch_lengths(
            self.branch_lengths_kernel
        ).numpy()
        self.tree_handler.branch_lengths[num_leaves:] = new_anc_branch_lengths

        if model_indices is not None:
            self.tree_handler.branch_lengths = (
                self.tree_handler.branch_lengths[..., model_indices]
            )

        emitter_copy = TreeEmitter(
            tree_handler = self.tree_handler,
            emission_init = sub_emission_init,
            insertion_init = sub_insertion_init,
            prior = self.prior,
            tree_loss_weight = self.tree_loss_weight,
            branch_lengths_init = None, # already set in the tree handler
            perturbate_cluster_index_prob = self.perturbate_cluster_index_prob,
            propagate_root = self.propagate_root
        ) 
        emitter_copy.num_models = len(model_indices)
        if share_kernels:
            emitter_copy.emission_kernel = self.emission_kernel
            emitter_copy.insertion_kernel = self.insertion_kernel
            emitter_copy.built = True
        return emitter_copy
    

    def get_config(self):
        config = super(TreeEmitter, self).get_config()
        config.update({
        "tree": self.tree_handler.to_newick(),
        "tree_loss_weight": self.tree_loss_weight,
        "propagate_root": self.propagate_root,
        "branch_lengths_init": self.branch_lengths_init,
        })
        return config
    

    @classmethod
    def from_config(cls, config):
        config["tree_handler"] = tensortree.TreeHandler.from_newick(config["tree"])
        config["branch_lengths_init"] = deserialize(config["branch_lengths_init"])
        del config["tree"]
        return super(TreeEmitter, cls).from_config(config)
    

    def __repr__(self):
        return f"TreeEmitter()"
    



    

def _make_default_rate_matrix(num_models=1):
    R, p = tensortree.substitution_models.LG()
    R = R[tf.newaxis]
    p = p[tf.newaxis]
    Q = tensortree.backend.make_rate_matrix(R, p)
    return Q, p



class PerturbationProbCallback(tf.keras.callbacks.Callback):

    def __init__(self, tree_layer, decay = 0.02, init_prob=0.0, min_prob=0.0):
        super(PerturbationProbCallback, self).__init__()
        self.tree_layer = tree_layer
        self.decay = decay
        self.init_prob = init_prob
        self.min_prob = min_prob

    def on_train_begin(self, logs=None):
        self.tree_layer.perturbate_cluster_index_prob.assign(self.init_prob)

    def on_train_batch_end(self, batch, logs=None):
        self.tree_layer.perturbate_cluster_index_prob.assign_add(-self.decay)
        if self.tree_layer.perturbate_cluster_index_prob < self.min_prob:
            self.tree_layer.perturbate_cluster_index_prob.assign(self.min_prob)
