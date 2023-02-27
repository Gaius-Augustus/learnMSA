import os
import tensorflow as tf
import learnMSA.msa_hmm.DirichletMixture as dm


class AminoAcidPrior(tf.keras.layers.Layer):
    """The default dirichlet mixture prior for amino acids.

    Args:
        comp_count: The number of components in the mixture.
        epsilon: A small constant for numerical stability.
    """
    def __init__(self, comp_count=1, epsilon=1e-16, **kwargs):
        super(AminoAcidPrior, self).__init__(**kwargs)
        self.comp_count = comp_count
        self.epsilon = epsilon
        
    def load(self, dtype):
        prior_path = os.path.dirname(__file__)+"/trained_prior/"
        dtype = dtype if isinstance(dtype, str) else dtype.name
        model_path = prior_path+"_".join([str(self.comp_count), "True", dtype, "_dirichlet/ckpt"])
        model = dm.load_mixture_model(model_path, self.comp_count, 20, trainable=False, dtype=dtype)
        self.emission_dirichlet_mix = model.layers[-1]
        
    def call(self, B, lengths):
        """Computes log pdf values for each match state.
        Args:
        B: A stack of k emission matrices. Assumes that the 20 std amino acids are the first 20 positions of the last dimension
            and that the states are ordered the standard way as seen in MsaHmmCell. Shape: (k, q, s)
        Returns:
            A tensor with the log pdf values of this Dirichlet mixture. The models can vary in length. 
            Shorter models will have a zero padding in the output. Shape: (k, max_model_length)
        """
        k = tf.shape(B)[0]
        max_model_length = tf.reduce_max(lengths)
        s = tf.shape(B)[2]
        #add a small constant to avoid underflow and division by 0 in the next line after
        B = B[:,1:max_model_length+1,:20] + self.epsilon 
        B /= tf.reduce_sum(B, axis=-1, keepdims=True) 
        B = tf.reshape(B, (-1, 20))
        prior = self.emission_dirichlet_mix.log_pdf(B)
        prior = tf.reshape(prior, (k, max_model_length))
        prior *= tf.cast(tf.sequence_mask(lengths), B.dtype)
        return prior 
    
    def get_config(self):
        config = super(AminoAcidPrior, self).get_config()
        cell_config = {
             "comp_count" : self.comp_count,
             "epsilon" : self.epsilon
        }
        config.update(cell_config)
        return config
    
    def __repr__(self):
        return f"AminoAcidPrior(comp_count={self.comp_count})"
    
    
class NullPrior(tf.keras.layers.Layer):
    """NullObject if no prior should be used.
    """
    def load(self, dtype):
        pass
    
    def call(self, B):
        k = tf.shape(B)[0]
        model_length = int((tf.shape(B)[1]-2)/2)
        return tf.zeros((k, model_length), dtype=B.dtype)
    
    def __repr__(self):
        return f"NullPrior()"
    
    
class ProfileHMMTransitionPrior(tf.keras.layers.Layer):
    """The default dirichlet mixture prior for profileHMM transitions.

    Args:
        match_comp: The number of components for the match prior.
        insert_comp: The number of components for the insert prior.
        delete_comp: The number of components for the delete prior.
        alpha_flank: Favors high probability of staying one of the 3 flanking states.
        alpha_single: Favors high probability for a single main model hit (avoid looping paths).
        alpha_frag: Favors models with high prob. to enter at the first match and exit after the last match.
        epsilon: A small constant for numerical stability.
    """
    def __init__(self, 
                 match_comp=1, 
                 insert_comp=1, 
                 delete_comp=1,
                 alpha_flank = 7000,
                 alpha_single = 1e9,
                 alpha_frag = 1e4,
                 epsilon=1e-16, 
                 **kwargs):
        super(ProfileHMMTransitionPrior, self).__init__(**kwargs)
        self.match_comp = match_comp
        self.insert_comp = insert_comp
        self.delete_comp = delete_comp
        #equivalent to the alpha parameters of a dirichlet mixture -1 .
        #these values are crutial when jointly optimizing the main model with the additional
        #"Plan7" states and transitions
        self.alpha_flank = alpha_flank
        self.alpha_single = alpha_single
        self.alpha_frag = alpha_frag
        self.epsilon = epsilon
        
    def load(self, dtype):
        prior_path = os.path.dirname(__file__)+"/trained_prior/transition_priors/"

        match_model_path = prior_path + "_".join(["match_prior", str(self.match_comp), dtype]) + "/ckpt"
        match_model = dm.load_mixture_model(match_model_path, self.match_comp, 3, trainable=False, dtype=dtype)
        self.match_dirichlet = match_model.layers[-1]
        insert_model_path = prior_path + "_".join(["insert_prior", str(self.insert_comp), dtype]) + "/ckpt"
        insert_model = dm.load_mixture_model(insert_model_path, self.insert_comp, 2, trainable=False, dtype=dtype)
        self.insert_dirichlet = insert_model.layers[-1]
        delete_model_path = prior_path + "_".join(["delete_prior", str(self.delete_comp), dtype]) + "/ckpt"
        delete_model = dm.load_mixture_model(delete_model_path, self.delete_comp, 2, trainable=False, dtype=dtype)
        self.delete_dirichlet = delete_model.layers[-1]
        
    def call(self, probs_list, flank_init_prob):
        """Computes log pdf values for each transition prior.
        Args:
        probs_list: A list of dictionaries that map transition type to probabilities per model.
        flank_init_prob: Flank init probabilities per model.
        Returns:
            A dictionary that maps prior names to lists of prior values per model.
        """
        match_dirichlet = []
        insert_dirichlet = []
        delete_dirichlet = []
        flank_prior = []
        hit_prior = []
        enex_prior = []
        for i,probs in enumerate(probs_list):
            log_probs = {part_name : tf.math.log(p) for part_name, p in probs.items()}
            #match state transitions
            p_match = tf.stack([probs["match_to_match"],
                                probs["match_to_insert"],
                                probs["match_to_delete"][1:]], axis=-1) + 1e-16
            p_match_sum = tf.reduce_sum(p_match, axis=-1, keepdims=True)
            match_dirichlet.append( tf.reduce_sum(self.match_dirichlet.log_pdf(p_match / p_match_sum)) )
            #insert state transitions
            p_insert = tf.stack([probs["insert_to_match"],
                               probs["insert_to_insert"]], axis=-1)
            insert_dirichlet.append( tf.reduce_sum(self.insert_dirichlet.log_pdf(p_insert)) )
            #delete state transitions
            p_delete = tf.stack([probs["delete_to_match"][:-1],
                               probs["delete_to_delete"]], axis=-1)
            delete_dirichlet.append( tf.reduce_sum(self.delete_dirichlet.log_pdf(p_delete)) )
            #other transitions
            flank = self.alpha_flank * log_probs["unannotated_segment_loop"] #todo: handle as extra case?
            flank += self.alpha_flank * log_probs["right_flank_loop"]
            flank += self.alpha_flank * log_probs["left_flank_loop"]
            flank += self.alpha_flank * log_probs["end_to_right_flank"]
            flank += self.alpha_flank * tf.math.log(flank_init_prob[i])
            flank_prior.append(tf.squeeze(flank))
            #uni-hit
            hit = self.alpha_single * tf.math.log(probs["end_to_right_flank"] + probs["end_to_terminal"]) 
            hit_prior.append(tf.squeeze(hit))
            #uniform entry/exit prior
            #rescale begin_to_match to sum to 1
            div = tf.math.maximum(self.epsilon, 1- probs["match_to_delete"][0]) 
            btm = probs["begin_to_match"] / div
            enex = tf.expand_dims(btm, 1) * tf.expand_dims(probs["match_to_end"], 0)
            enex = tf.linalg.band_part(enex, 0, -1)
            enex = tf.math.log(tf.math.maximum(self.epsilon, 1 - enex))
            enex_prior.append( self.alpha_frag * (tf.reduce_sum(enex) - enex[0, -1]) )
        prior_val = {
            "match_prior" : match_dirichlet,
            "insert_prior" : insert_dirichlet,
            "delete_prior" : delete_dirichlet,
            "flank_prior" : flank_prior,
            "hit_prior" : hit_prior,
            "enex_prior" : enex_prior}
        prior_val = {k : tf.stack(v) for k,v in prior_val.items()}
        return prior_val
    
    def get_config(self):
        config = super(ProfileHMMTransitionPrior, self).get_config()
        cell_config = {
             "match_comp" : self.match_comp, 
             "insert_comp" : self.insert_comp, 
             "delete_comp" : self.delete_comp, 
             "alpha_flank" : self.alpha_flank, 
             "alpha_single" : self.alpha_single, 
             "alpha_frag" : self.alpha_frag, 
             "epsilon" : self.epsilon
        }
        config.update(cell_config)
        return config
    
    def __repr__(self):
        return f"ProfileHMMTransitionPrior(match_comp={self.match_comp}, insert_comp={self.insert_comp}, delete_comp={self.delete_comp}, alpha_flank={self.alpha_flank}, alpha_single={self.alpha_single}, alpha_frag={self.alpha_frag})"

    
