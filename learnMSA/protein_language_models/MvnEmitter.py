import tensorflow as tf
import numpy as np
import os
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset
from learnMSA.msa_hmm.Emitter import ProfileHMMEmitter
import learnMSA.msa_hmm.Initializers as Initializers
import learnMSA.msa_hmm.Priors as priors
from learnMSA.protein_language_models.BilinearSymmetric import make_scoring_model
from learnMSA.protein_language_models.MvnMixture import MvnMixture
from learnMSA.protein_language_models.MvnPrior import MvnPrior, InverseGammaPrior, get_expected_emb
import learnMSA.protein_language_models.Common as Common
from learnMSA.msa_hmm.Utility import DefaultDiagBijector, deserialize




def make_joint_prior(scoring_model_config : Common.ScoringModelConfig, num_prior_components,
                    inv_gamma_alpha, inv_gamma_beta, dtype):
    prior_list = [priors.AminoAcidPrior(dtype=dtype),
                MvnPrior(scoring_model_config, num_prior_components, dtype=dtype),
                InverseGammaPrior(inv_gamma_alpha, inv_gamma_beta),
                priors.NullPrior()]
    num_aa = len(SequenceDataset.alphabet)
    kernel_split = [num_aa, num_aa + scoring_model_config.dim, num_aa + 2*scoring_model_config.dim]
    return priors.JointEmissionPrior(prior_list, kernel_split, dtype=dtype)


#have a single emitter that handles both AA inputs and embeddings
class MvnEmitter(ProfileHMMEmitter):
    def __init__(self,
                 scoring_model_config : Common.ScoringModelConfig,
                 emission_init=None,
                 insertion_init=None,
                 num_prior_components = Common.PRIOR_DEFAULT_COMPONENTS,
                 diag_init_var = 1.,
                 full_covariance = False,
                 temperature = 10.,
                 inv_gamma_alpha = 3.,
                 inv_gamma_beta = 3.,
                 regularize_variances = False, #deprecated
                 dtype=tf.float32,
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
            prior = make_joint_prior(scoring_model_config, num_prior_components, inv_gamma_alpha, inv_gamma_beta, dtype)

        super(MvnEmitter, self).__init__(emission_init, insertion_init, prior, dtype=dtype, **kwargs)

        self.diag_init_var = diag_init_var
        self.full_covariance = full_covariance
        self.temperature = temperature
        self.inv_gamma_alpha = inv_gamma_alpha
        self.inv_gamma_beta = inv_gamma_beta
        self.regularize_variances = regularize_variances
        self.num_aa = len(SequenceDataset.alphabet)-1


    def build(self, input_shape):
        if self.full_covariance:
            s = self.num_aa + self.scoring_model_config.dim + self.scoring_model_config.dim * (self.scoring_model_config.dim+1) // 2
        else:
            s = self.num_aa + 2*self.scoring_model_config.dim
        shape = (input_shape[0], s+1)
        super().build(shape)


    def make_B(self):
        #first use the base class logic to construct a prototype B
        #this B does not contain the correct variances for the MVN disributions yet
        B = super(MvnEmitter, self).make_B()
        max_model_len = max(self.lengths)
        # make use of the shared insertions and compute emissions for all matches and one insertion
        B_mvn_param = B[:, :max_model_len+1, self.num_aa+1:]
        # create the mvn mixture object in each step with the current kernel values
        self.mvn_mixture = MvnMixture(self.scoring_model_config.dim, B_mvn_param[..., tf.newaxis, :],
                                        diag_only = not self.full_covariance,
                                        diag_bijector = DefaultDiagBijector(self.diag_init_var))
        cov = self.mvn_mixture.component_covariances()
        if self.full_covariance:
            raise ValueError("Full covariance matrix is currently not fully implemented.")
        else:
            #remove mixture component dimension (not used)
            var = cov[...,0,:]
            #append copies of insertion states
            ins_copy = tf.repeat(var[:,:1], repeats=max_model_len+2, axis=1)
            var = tf.concat([var, ins_copy], axis=1)
            # replace the kernel values for the variances in B with the actual variances
            B_no_var = B[..., :self.num_aa+1+self.scoring_model_config.dim]
            B = tf.concat([B_no_var, var], axis=-1)
        return B


    def make_emission_matrix(self, i):
        aa_em = self.emission_kernel[i][:, :self.num_aa]
        emb_em = self.emission_kernel[i][:, self.num_aa:]
        #drop the terminal state probs since we have identical ones for the embeddings
        aa_emissions = self.make_emission_matrix_from_kernels(aa_em, self.insertion_kernel[i][:self.num_aa], self.lengths[i])
        """ Construct the emission matrix the same way as usual but leave away the softmax.
        """
        emb_ins = self.insertion_kernel[i][self.num_aa:]
        emb_em = tf.concat([tf.expand_dims(emb_ins, 0),
                            emb_em,
                            tf.stack([emb_ins]*(self.lengths[i]+1))] , axis=0)
        end_state_emission = tf.zeros_like(emb_em[:1])
        emb_emissions = tf.concat([emb_em, end_state_emission], axis=0)
        return tf.concat([aa_emissions, emb_emissions], -1)


    def compute_embedding_emission_probs(self, emb_inputs, training=False):
        # for convenient reshaping
        _, batch, seq_len, _  = tf.unstack(tf.shape(emb_inputs))
        #compute emission density value
        log_pdf = self.mvn_mixture.log_pdf(tf.reshape(emb_inputs[..., :-1], (self.num_models, batch*seq_len, self.scoring_model_config.dim)))
        log_pdf = tf.reshape(log_pdf, (self.num_models, batch, seq_len, max(self.lengths)+1))

        # we optimized away some states and have to append clones of the first insert to make activations like softmax work correctly
        rest = tf.repeat(log_pdf[..., :1], tf.shape(self.B_transposed)[-1] - max(self.lengths)-1, axis=-1)
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
        return emission_probs


    def get_aux_loss(self):
        if self.regularize_variances:
            return 0.01 * self.mvn_mixture.get_regularization_L2_loss()
        else:
            return 0.


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
                                inv_gamma_alpha = self.inv_gamma_alpha,
                                inv_gamma_beta = self.inv_gamma_beta,
                                dtype = self.dtype)
        if share_kernels:
            emitter_copy.emission_kernel = self.emission_kernel
            emitter_copy.insertion_kernel = self.insertion_kernel
            emitter_copy.built = True
        return emitter_copy


    def get_config(self):
        config = super(MvnEmitter, self).get_config()
        config.update({"scoring_model_config" : self.scoring_model_config.to_dict(),
                        "num_prior_components" : self.num_prior_components,
                        "diag_init_var" : self.diag_init_var,
                        "full_covariance" : self.full_covariance,
                        "temperature" : self.temperature,
                        "inv_gamma_alpha" : self.inv_gamma_alpha,
                        "inv_gamma_beta" : self.inv_gamma_beta})
        return config


    @classmethod
    def from_config(cls, config):
        config["emission_init"] = [deserialize(e) for e in config["emission_init"]]
        config["insertion_init"] = [deserialize(i) for i in config["insertion_init"]]
        config["prior"] = deserialize(config["prior"])
        config["scoring_model_config"] = Common.ScoringModelConfig(**config["scoring_model_config"])
        lengths = config.pop("lengths")
        emitter = cls(**config)
        emitter.set_lengths(lengths)
        return emitter


    def __repr__(self):
        parent = super(MvnEmitter, self).__repr__()
        return f"MvnEmitter(scoring_model_config = {self.scoring_model_config}, {parent})"



class AminoAcidPlusMvnEmissionInitializer(Initializers.EmissionInitializer):
    """ Initializes emission kernels for joint, conditionally independent
    amino acid and multivariate normal distributions.
    """

    def __init__(self,
                 scoring_model_config : Common.ScoringModelConfig,
                 dist=None,
                 aa_dist=np.log(Initializers.background_distribution),
                 num_prior_components=Common.PRIOR_DEFAULT_COMPONENTS,
                 scale_kernel_init = Initializers.RandomNormalInitializer(stddev=0.02),
                 full_covariance=False):
        self.aa_dist = aa_dist
        self.scoring_model_config = scoring_model_config
        self.num_prior_components = num_prior_components
        self.scale_kernel_init = scale_kernel_init
        self.full_covariance = full_covariance
        if dist is None:
            self.expected_emb = get_expected_emb(
                scoring_model_config, num_prior_components
            )
            dist = np.concatenate([self.aa_dist, self.expected_emb], axis=0)
        super(AminoAcidPlusMvnEmissionInitializer, self).__init__(dist = dist)


    def __call__(self, shape, dtype=tf.float32, **kwargs):
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


    def get_config(self):  # To support serialization
        config = super(AminoAcidPlusMvnEmissionInitializer, self).get_config()
        config.update({
            "scoring_model_config" : self.scoring_model_config.to_dict(),
            "aa_dist" : self.aa_dist.tolist(),
            "num_prior_components" : self.num_prior_components,
            "scale_kernel_init" : self.scale_kernel_init,
            "full_covariance" : self.full_covariance,
            "expected_emb" : self.expected_emb.tolist()
        })
        return config

    @classmethod
    def from_config(cls, config):
        config["scoring_model_config"] = Common.ScoringModelConfig(**config["scoring_model_config"])
        config["aa_dist"] = np.array(config["aa_dist"])
        config["dist"] = np.array(config["dist"])
        expected_emb = np.array(config["expected_emb"])
        del config["expected_emb"]
        instance = cls(**config)
        instance.expected_emb = expected_emb
        return instance



tf.keras.utils.get_custom_objects()["MvnEmitter"] = MvnEmitter
tf.keras.utils.get_custom_objects()["AminoAcidPlusMvnEmissionInitializer"] = AminoAcidPlusMvnEmissionInitializer