from learnMSA.msa_hmm.Transitioner import ProfileHMMTransitioner
import learnMSA.msa_hmm.Initializers as initializers
import tensorflow as tf
import numpy as np
import tensortree 

tensortree.set_backend("tensorflow")


class ClusterTransitioner(ProfileHMMTransitioner):
    """
    A transitioner that allows different cluster of transitioners within one model.
    """
    def __init__(self,
                 num_clusters,
                 cluster_indices,
                transition_init = initializers.make_default_transition_init(),
                flank_init = initializers.make_default_flank_init(),
                prior = None,
                frozen_kernels={},
                **kwargs):
        super(ClusterTransitioner, self).__init__(transition_init, flank_init, prior, frozen_kernels, **kwargs)
        self.num_clusters = num_clusters
        self.cluster_indices = cluster_indices

    
    def get_kernel_shape(self, base_shape):
        return (self.num_clusters,) + base_shape
    

    def call(self, inputs):
        """ 
        Args: 
                inputs: A tensor of shape (k, b, q) 
        Returns:
                Shape (k, b, q)
        """

        tf.debugging.assert_equal(tf.shape(inputs)[:2], 
                                  tf.shape(self.indices)[:2],
                                 message=("The first two dimensions of inputs and "
                                           + "indices must be equal."))

        # get A_t or A of shape (k, num_clusters, q, q)
        A = self.A if self.reverse else self.A_t

        # we have to select the correct parameters for each input sequence
        cluster_indices = tf.gather(self.cluster_indices, self.indices)
        A = tf.gather(A, cluster_indices, batch_dims=1)

        return tf.linalg.matvec(A, inputs)


    def make_initial_distribution(self, indices=None):
        """Constructs the initial state distribution per model which depends on the transition probabilities.
        Args:
            indices: A tensor of shape (k, b) that contains the index of each input sequence.
        Returns:
            A probability distribution per model. Shape: (k,) + get_kernel_shape((q,)) if indices is None
            or (k, b, q) if indices is not None.
        """
        init_dists = super(ClusterTransitioner, self).make_initial_distribution(indices)
        cluster_indices = tf.gather(self.cluster_indices, self.indices)
        init_dists = tf.gather(init_dists, cluster_indices, batch_dims=1)
        return init_dists


class TreeTransitioner(ClusterTransitioner):
    """
    A transitioner that allows different cluster of transitioners within one model.
    """
    def __init__(self,
                 tree_handler : tensortree.TreeHandler,
                transition_init = initializers.make_default_transition_init(),
                flank_init = initializers.make_default_flank_init(),
                prior = None,
                frozen_kernels={},
                **kwargs):
        cluster_indices = np.copy(tree_handler.get_parent_indices_by_height(0))
        cluster_indices -= tree_handler.num_leaves # indices are sorted by height
        num_clusters = np.unique(cluster_indices).size
        super(TreeTransitioner, self).__init__(num_clusters,
                                                cluster_indices,
                                                transition_init, 
                                                flank_init, 
                                                prior, 
                                                frozen_kernels, 
                                                **kwargs)
        