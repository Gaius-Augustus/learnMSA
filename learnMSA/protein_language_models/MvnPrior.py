import os
import tensorflow as tf
import learnMSA.protein_language_models.Common as Common
from learnMSA.protein_language_models.MvnMixture import MvnMixture, DefaultDiagBijector


class MvnPrior(tf.keras.layers.Layer):
    """ A multivariate normal prior over embeddings from a scoring model.
    """
    def __init__(self, 
                 scoring_model_config : Common.ScoringModelConfig,
                 num_components=Common.PRIOR_DEFAULT_COMPONENTS, 
                 **kwargs):
        super(MvnPrior, self).__init__(**kwargs)
        self.scoring_model_config = scoring_model_config
        self.num_components = num_components


    def load(self, dtype):
        # load the underlying scoring model
        self.prior_path = Common.get_prior_path(self.scoring_model_config, self.num_components)
        self.prior_path = os.path.dirname(__file__)+f"/../protein_language_models/"+self.prior_path
        # load a precomputed prior
        self.pdf_model = make_pdf_model(self.scoring_model_config, 
                                    num_components=self.num_components,
                                    trainable=False,
                                    aggregate_result=False)
        self.pdf_model.load_weights(self.prior_path)
        self.pdf_model.trainable = False
        self.mvn_prior_layer = get_mvn_layer(self.pdf_model)
        self.mvn_prior_layer.trainable = False

 
    def call(self, B, lengths):
        """L2 regularization for each match state.
        Args:
        B: A stack of k emission matrices. Shape: (k, q, s)
        Returns:
        A tensor with the L2 regularization values. Shape: (k, q)
        """
        max_model_length = tf.reduce_max(lengths)
        length_mask = tf.cast(tf.sequence_mask(lengths), B.dtype)
        # make sure padding is zero
        match_states = B[:,1:max_model_length+1]
        # compute the prior
        mvn_log_pdf = self.pdf_model(match_states)
        mvn_log_pdf *= length_mask
        return mvn_log_pdf


    def get_config(self):
        config = super(MvnPrior, self).get_config()
        config.update({
             "scoring_model_config" : self.scoring_model_config,
            "num_components" : self.num_components
        })
        return config



class MvnPriorLayer(tf.keras.layers.Layer):
    """ Utility class that mediates between the actual prior class and the keras model used for pretraining.
    """
    def __init__(self, 
                 scoring_model_config : Common.ScoringModelConfig,
                 num_components=Common.PRIOR_DEFAULT_COMPONENTS, 
                 kernel_init = tf.random_normal_initializer(stddev=0.01),
                 mixture_kernel_init = tf.random_normal_initializer(stddev=0.01),
                 diag_init_var = 1., 
                 full_covariance = False,
                 trainable=True,
                 **kwargs):
        super(MvnPriorLayer, self).__init__(**kwargs)
        self.scoring_model_config = scoring_model_config
        self.num_components = num_components
        self.kernel_init = kernel_init
        self.mixture_kernel_init = mixture_kernel_init
        self.diag_init_var = diag_init_var
        self.full_covariance = full_covariance
        self.trainable = trainable

        
    def build(self, input_shape):
        # build the underlying parameter kernel closely related to the mvn parameters
        d = self.scoring_model_config.dim
        kernel_size = d + d * (d+1) // 2 if self.full_covariance else 2*d
        self.kernel = self.add_weight(name="kernel", 
                                    shape=(1, 1, self.num_components, kernel_size), 
                                    initializer=self.kernel_init,
                                    trainable=self.trainable)
        if self.num_components > 1:
            self.mixture_coeff_kernel = self.add_weight(name="mixture_coeff_kernel",
                                                        shape=(1, 1, self.num_components),
                                                        initializer=self.mixture_kernel_init,
                                                        trainable=self.trainable)
        else:
            self.mixture_coeff_kernel = None


    def get_mixture(self):
        # build the mvn mixture
        return MvnMixture(self.scoring_model_config.dim, 
                                    self.kernel, 
                                    self.mixture_coeff_kernel,
                                    diag_only = not self.full_covariance,
                                    diag_bijector = DefaultDiagBijector(self.diag_init_var),
                                    precomputed = not self.trainable)


    def call(self, inputs):
        """
        Args:
            inputs: Shape (n, m, dim)
        Returns:
            Shape (n, m)
        """
        # build the mvn mixture
        return self.get_mixture().log_pdf(inputs)[..., 0]


    def get_config(self):
        config = super(MvnPriorLayer, self).get_config()
        config.update({
             "scoring_model_config" : self.scoring_model_config,
            "num_components" : self.num_components,
            "kernel_init" : self.kernel_init,
            "mixture_kernel_init" : self.mixture_kernel_init,
            "diag_init_var" : self.diag_init_var,
            "full_covariance" : self.full_covariance,
            "trainable" : self.trainable
        })
        return config



def make_pdf_model(scoring_model_config : Common.ScoringModelConfig, 
                   num_components=Common.PRIOR_DEFAULT_COMPONENTS, 
                   trainable=True, 
                   aggregate_result=False):
    """ Utility that constructs a keras model around the prior. Can be used for pretraining and easy saving/loading.
        Args:
            scoring_model_config: The scoring model configuration. Used to load the correct scoring model.
            num_components: The number of mixture components.
            trainable: If True, the prior parameters are trainable. When False, heavy computations like 
                        computing the pseudo inverse of the scale matrix are done once and stored in the layer.
            aggregate_result: If True, the result is averaged over sequences and batch. Otherwise the model output has shape (batch, num_states).
    """
    embeddings = tf.keras.Input((None, scoring_model_config.dim))
    # compute log pdf per observation
    log_pdf = MvnPriorLayer(scoring_model_config, num_components, trainable=trainable)(embeddings)
    # zero out pdfs of zero embeddings (assumed padding)
    mask = tf.reduce_any(tf.not_equal(embeddings, 0), -1)
    mask = tf.cast(mask, log_pdf.dtype)
    if aggregate_result:
        log_pdf = aggregate(log_pdf, mask)
    else:
        log_pdf *= mask
    model = tf.keras.Model(inputs=[embeddings], outputs=[log_pdf])
    return model



def get_mvn_layer(pdf_model):
    mvn_layer = None
    for layer in pdf_model.layers:
        if isinstance(layer, MvnPriorLayer):
            mvn_layer = layer
            break
    return mvn_layer



def aggregate(x, mask):
    """ Utility that reduces values to a scalar by averaging over sequences and batch.
        Args:
            x: A tensor of shape (batch, seq_len)
            mask: A tensor of same shape and type as x, indicating non-padding positions.  
    """
    seq_lens = tf.reduce_sum(mask, -1)
    # average per sequence
    seg_avg = tf.reduce_sum(x * mask, -1) / tf.maximum(seq_lens, 1.)
    # average over batch
    return tf.reduce_mean(seg_avg)


emb_cache = {}

def get_expected_emb(scoring_model_config : Common.ScoringModelConfig, num_prior_components):
    if num_prior_components == 0:
        return np.random.normal(scale=0.02, size=scoring_model_config.dim)
    else:
        prior_weight_path = Common.get_prior_path(scoring_model_config, num_prior_components)
        if prior_weight_path not in emb_cache:
            # load the prior model
            pdf_model = make_pdf_model(scoring_model_config, num_prior_components, trainable=False)
            pdf_model.load_weights(os.path.dirname(__file__)+f"/../protein_language_models/"+prior_weight_path)
            emb_cache[prior_weight_path] = get_mvn_layer(pdf_model).get_mixture().expectation()[0,0].numpy()
        return emb_cache[prior_weight_path]


tf.keras.utils.get_custom_objects()["MvnPriorLayer"] = MvnPriorLayer
tf.keras.utils.get_custom_objects()["MvnPrior"] = MvnPrior