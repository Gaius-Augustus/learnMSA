from learnMSA.msa_hmm.Transitioner import ProfileHMMTransitioner
import learnMSA.msa_hmm.Initializers as initializers



class TreeTransitioner(ProfileHMMTransitioner):
    """
    A transitioner that allows different cluster of transitioners within one model.
    """
    def __init__(self,
                 num_clusters,
                transition_init = initializers.make_default_transition_init(),
                flank_init = initializers.make_default_flank_init(),
                prior = None,
                frozen_kernels={},
                **kwargs):
        super(TreeTransitioner, self).__init__(transition_init, flank_init, prior, frozen_kernels, **kwargs)
        self.num_clusters = num_clusters

    
    def get_kernel_shape(self, base_shape):
        return (self.num_clusters,) + base_shape