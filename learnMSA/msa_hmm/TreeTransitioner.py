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
    

    def recurrent_init(self, indices=None):
        super(ClusterTransitioner, self).recurrent_init(indices)

        if indices is not None:
            # we have to select the correct parameters for each input sequence
            self.A = self.make_sample_A(self.indices, self.A)
            self.A_t = self.make_sample_A(self.indices, self.A_t)
    

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

        A = self.A_t if self.reverse else self.A

        return tf.einsum("kbq,kbqz->kbz", inputs, A)
        #return tf.linalg.matvec(A, inputs)
    

    def make_sample_A(self, indices, A):
        """Constructs the transition probabilities per model and sample which depends on the cluster indices.
        Args:
            indices: A tensor of shape (k, b) that contains the index of each sample.
            A: A tensor of shape (k, c, q, q) that contains the transition probabilities 
                of each of c clusters in each of the k models.
        Returns:
            A transition matrix per model and sample. Shape: (k,b,q,q)
        """
        cluster_indices = tf.gather(self.cluster_indices, indices)
        A = tf.gather(A, cluster_indices, batch_dims=1)
        return A


    def make_initial_distribution(self, indices):
        """Constructs the initial state distribution per model which depends on the transition probabilities.
        Args:
            indices: A tensor of shape (k, b) that contains the index of each input sequence.
        Returns:
            A probability distribution per model. Shape: (k,b,q)
        """
        init_dists = super(ClusterTransitioner, self).make_initial_distribution(indices)
        init_dists = init_dists[:,0]
        cluster_indices = tf.gather(self.cluster_indices, indices)
        init_dists = tf.gather(init_dists, cluster_indices, batch_dims=1)
        return init_dists
    

    def duplicate(self, model_indices=None, share_kernels=False):
        if model_indices is None:
            model_indices = range(len(self.transition_init))
        sub_transition_init = []
        sub_flank_init = []
        for i in model_indices:
            transition_init_dict = {key : tf.constant_initializer(kernel.numpy())
                                       for key, kernel in self.transition_kernel[i].items()}
            sub_transition_init.append(transition_init_dict)
            sub_flank_init.append(tf.constant_initializer(self.flank_init_kernel[i].numpy()))
        transitioner_copy = ClusterTransitioner(
                                        num_clusters=self.num_clusters,
                                        cluster_indices=self.cluster_indices,
                                        transition_init = sub_transition_init,
                                        flank_init = sub_flank_init,
                                        prior = self.prior,
                                        frozen_kernels = self.frozen_kernels,
                                        dtype = self.dtype) 
        if share_kernels:
            transitioner_copy.transition_kernel = self.transition_kernel
            transitioner_copy.flank_init_kernel = self.flank_init_kernel
            transitioner_copy.built = True
        return transitioner_copy
    

    def get_config(self):
        config = super(ClusterTransitioner, self).get_config()
        config.update({
            'num_clusters': self.num_clusters,
            'cluster_indices': self.cluster_indices,
        })
        return config
    


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