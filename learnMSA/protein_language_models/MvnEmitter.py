import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset
from learnMSA.msa_hmm.Emitter import ProfileHMMEmitter
import learnMSA.msa_hmm.Initializers as Initializers
import learnMSA.msa_hmm.Priors as priors
from learnMSA.protein_language_models.BilinearSymmetric import make_scoring_model
from learnMSA.protein_language_models.MvnMixture import MvnMixture, DefaultDiagBijector
from learnMSA.protein_language_models.MvnPrior import MvnPrior, get_expected_emb
import learnMSA.protein_language_models.Common as Common




def make_joint_prior(scoring_model_config : Common.ScoringModelConfig, num_prior_components):
    prior_list = [priors.AminoAcidPrior(),
                MvnPrior(scoring_model_config, num_prior_components),
                priors.NullPrior()]
    num_aa = len(SequenceDataset.alphabet)-1
    kernel_split = [num_aa, num_aa + scoring_model_config.dim]
    return priors.JointEmissionPrior(prior_list, kernel_split)


#have a single emitter that handles both AA inputs and embeddings
class MvnEmitter(ProfileHMMEmitter):
    def __init__(self, 
                 scoring_model_config : Common.ScoringModelConfig,
                 emission_init=None, 
                 insertion_init=None,
                 num_prior_components = Common.PRIOR_DEFAULT_COMPONENTS,
                 diag_init_var = 1,
                 full_covariance = False,
                 temperature = 10.,
                 regularize_variances = True,
                 **kwargs):

        self.scoring_model_config = scoring_model_config
        self.num_prior_components = num_prior_components

        if emission_init is None:
            emission_init = AminoAcidPlusMvnEmissionInitializer(scoring_model_config=scoring_model_config, 
                                                                            num_prior_components=num_prior_components,
                                                                            full_covariance=full_covariance)
        if insertion_init is None:
            insertion_init = AminoAcidPlusMvnEmissionInitializer(scoring_model_config=scoring_model_config, 
                                                                            num_prior_components=num_prior_components,
                                                                            full_covariance=full_covariance)

        if "prior" in kwargs:
            prior = kwargs["prior"]
            del kwargs["prior"]
        else:
            prior = make_joint_prior(scoring_model_config, num_prior_components)
                                                             
        super(MvnEmitter, self).__init__(emission_init, insertion_init, prior, **kwargs)

        self.diag_init_var = diag_init_var
        self.full_covariance = full_covariance
        self.temperature = temperature
        self.regularize_variances = regularize_variances
        self.num_aa = len(SequenceDataset.alphabet)-1


    def build(self, input_shape):
        if self.full_covariance:
            s = self.num_aa + self.scoring_model_config.dim + self.scoring_model_config.dim * (self.scoring_model_config.dim+1) // 2
        else:
            s = self.num_aa + 2*self.scoring_model_config.dim
        shape = (input_shape[0], s+1)
        super().build(shape)
    

    def recurrent_init(self):
        """ Automatically called before each recurrent run. Should be used for setups that
            are only required once per application of the recurrent layer.
        """
        super(MvnEmitter, self).recurrent_init()
        # make use of the shared insertions and compute emissions for all matches and one insertion
        B_mvn_param = self.B[:, :max(self.length)+1, self.num_aa+1:]
        # create the mvn mixture object in each step with the current kernel values
        self.mvn_mixture = MvnMixture(self.scoring_model_config.dim, B_mvn_param[..., tf.newaxis, :],
                                        diag_only = not self.full_covariance,
                                        diag_bijector = DefaultDiagBijector(self.diag_init_var))
                                    

    def make_emission_matrix(self, i):
        aa_em = self.emission_kernel[i][:, :self.num_aa]
        emb_em = self.emission_kernel[i][:, self.num_aa:]
        #drop the terminal state probs since we have identical ones for the embeddings
        aa_emissions = self.make_emission_matrix_from_kernels(aa_em, self.insertion_kernel[i][:self.num_aa], self.length[i])
        """ Construct the emission matrix the same way as usual but leave away the softmax.
        """
        emb_ins = self.insertion_kernel[i][self.num_aa:]
        emb_em = tf.concat([tf.expand_dims(emb_ins, 0), 
                            emb_em, 
                            tf.stack([emb_ins]*(self.length[i]+1))] , axis=0)
        end_state_emission = tf.zeros_like(emb_em[:1])
        emb_emissions = tf.concat([emb_em, end_state_emission], axis=0)
        return tf.concat([aa_emissions, emb_emissions], -1)
        
        
    def compute_embedding_emission_probs(self, emb_inputs, training=False):
        # for convenient reshaping
        _, batch, seq_len, _  = tf.unstack(tf.shape(emb_inputs))
        #compute emission density value
        log_pdf = self.mvn_mixture.log_pdf(tf.reshape(emb_inputs[..., :-1], (self.num_models, batch*seq_len, self.scoring_model_config.dim)))
        log_pdf = tf.reshape(log_pdf, (self.num_models, batch, seq_len, max(self.length)+1))

        # we optimized away some states and have to append clones of the first insert to make activations like softmax work correctly
        rest = tf.repeat(log_pdf[..., :1], tf.shape(self.B_transposed)[-1] - max(self.length)-1, axis=-1)
        log_pdf = tf.concat([log_pdf, rest], axis=-1)

        # compute density to the power of 1/temperature 
        pdf = tf.exp(log_pdf/(self.temperature * self.scoring_model_config.dim))
        pdf += 1e-20 #protect against underflow

        #padding/terminal states
        terminal_padding = emb_inputs[..., -1:]
        # insert zero for terminal positions, except...
        pdf *= 1-terminal_padding 
        # ...insert ones where sequence input is terminal and state is also terminal
        pdf += terminal_padding * self.B[:, tf.newaxis, tf.newaxis, :, self.num_aa]

        return pdf 
        
        
    def call(self, inputs, end_hints=None, training=False):
        """ 
        Args: 
            inputs: Shape (num_models, batch, seq_len, d) 
            end_hints: A tensor of shape (num_models, batch_size, 2, num_states) that contains the correct state for the left and right ends of each chunk.
        Returns:
            Shape (num_models, batch, seq_len, num_states)
        """
        aa_emission_probs = super().call(inputs[..., :self.num_aa+1], training=training)
        emb_emission_probs = self.compute_embedding_emission_probs(inputs[..., self.num_aa+1:], training=training)
        emission_probs = aa_emission_probs * emb_emission_probs 
        if self.regularize_variances:
            self.add_loss(0.01 * self.mvn_mixture.get_regularization_L2_loss())
        return emission_probs
    
    
    def duplicate(self, model_indices=None, share_kernels=False):
        if model_indices is None:
            model_indices = range(len(self.emission_init))
        sub_emission_init = [tf.constant_initializer(self.emission_kernel[i].numpy()) for i in model_indices]
        sub_insertion_init = [tf.constant_initializer(self.insertion_kernel[i].numpy()) for i in model_indices]
        #todo: this does not dublicate embedding insertion kernels which is probably ok
        emitter_copy = MvnEmitter(
                                scoring_model_config = self.scoring_model_config,  
                                emission_init = sub_emission_init,
                                insertion_init = sub_insertion_init,
                                num_prior_components = self.num_prior_components,
                                diag_init_var = self.diag_init_var,
                                full_covariance = self.full_covariance,
                                temperature = self.temperature,
                                frozen_insertions = self.frozen_insertions,
                                dtype = self.dtype) 
        if share_kernels:
            emitter_copy.emission_kernel = self.emission_kernel
            emitter_copy.insertion_kernel = self.insertion_kernel
            emitter_copy.built = True
        return emitter_copy

    
    def get_config(self):
        config = super(MvnEmitter, self).get_config()
        config.update({"scoring_model_config" : self.scoring_model_config,
                        "num_prior_components" : self.num_prior_components,
                        "diag_init_var" : self.diag_init_var,
                        "full_covariance" : self.full_covariance,
                        "temperature" : self.temperature})
        return config


    def __repr__(self):
        parent = super(MvnEmitter, self).__repr__()
        return f"MvnEmitter(scoring_model_config = {self.scoring_model_config}, {parent})"



class AminoAcidPlusMvnEmissionInitializer(Initializers.EmissionInitializer):
    """ Initializes emission kernels for joint, conditionally independent amino acid and multivariate normal distributions.
    """

    def __init__(self,
                 scoring_model_config : Common.ScoringModelConfig,
                 aa_dist=np.log(Initializers.background_distribution), 
                 num_prior_components=100,
                 scale_kernel_init = tf.random_normal_initializer(stddev=0.02),
                 full_covariance=False):
        self.aa_dist = aa_dist
        self.scoring_model_config = scoring_model_config
        self.num_prior_components = num_prior_components
        self.expected_emb = get_expected_emb(scoring_model_config, num_prior_components)
        self.scale_kernel_init = scale_kernel_init
        self.full_covariance = full_covariance
        dist = np.concatenate([self.aa_dist, self.expected_emb], axis=0)
        super(AminoAcidPlusMvnEmissionInitializer, self).__init__(dist = dist)


    def __call__(self, shape, dtype=None, **kwargs):
        assert shape[-1] >= self.aa_dist.size
        emb_dim = self.expected_emb.shape[-1]
        if self.full_covariance:
            assert (shape[-1] - self.aa_dist.size) == emb_dim + emb_dim * (emb_dim+1) // 2, f"shape[-1]={shape[-1]} emb_dim={emb_dim}"
        else:
            assert (shape[-1] - self.aa_dist.size) == 2*emb_dim, f"shape[-1]={shape[-1]} emb_dim={emb_dim}"
        aa_plus_emb_init = super(AminoAcidPlusMvnEmissionInitializer, self).__call__(list(shape[:-1])+[self.aa_dist.size + emb_dim], dtype=dtype, **kwargs)
        if self.full_covariance:
            scale_init = self.scale_kernel_init(shape=list(shape[:-1])+[emb_dim * (emb_dim+1) // 2], dtype=dtype)
        else:
            scale_init = tf.zeros(shape=list(shape[:-1])+[emb_dim], dtype=dtype)
        return tf.concat([aa_plus_emb_init, scale_init], axis=-1)


        
tf.keras.utils.get_custom_objects()["MvnEmitter"] = MvnEmitter