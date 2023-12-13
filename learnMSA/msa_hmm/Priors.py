import os
import tensorflow as tf
import learnMSA.msa_hmm.DirichletMixture as dm
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset
#from learnMSA.protein_language_models.MultivariateNormalPrior import MultivariateNormalPrior, make_pdf_model
import learnMSA.protein_language_models.Common as Common
from learnMSA.protein_language_models.BilinearSymmetric import make_scoring_model


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
        config.update({
                    "comp_count" : self.comp_count,
                    "epsilon" : self.epsilon
                    })
        return config
    
    def __repr__(self):
        return f"AminoAcidPrior(comp_count={self.comp_count})"

    

class L2Regularizer(tf.keras.layers.Layer):
    """ A simple L2 regularizer for insertion and match state parameters.
    """
    def __init__(self, 
                 L2_match=10.0, 
                 L2_insert=0.,
                 use_shared_embedding_insertions=True, 
                 **kwargs):
        super(L2Regularizer, self).__init__(**kwargs)
        self.L2_match = L2_match
        self.L2_insert = L2_insert
        self.use_shared_embedding_insertions = use_shared_embedding_insertions
        
    def load(self, dtype):
        pass

    def get_l2_loss(self, B, lengths):
        max_model_length = tf.reduce_max(lengths)
        length_mask = tf.cast(tf.sequence_mask(lengths), B.dtype)
        #square all parameters
        B_sq = tf.math.square(B[...,len(SequenceDataset.alphabet)-1:-1])
        #reduce the embedding dimension
        B_sq = tf.reduce_sum(B_sq, -1)
        #regularization per match is just the sum of the respective squares 
        #(we will deal with non-match states in the end)
        reg_emb = B_sq[:,1:max_model_length+1]
        #depending on how we implemented insertions, we add a different insertion term to all matches
        if self.use_shared_embedding_insertions:
            #all insertions are the same, just use the first one 
            reg_ins = B_sq[:,:1]
        else:
            #insertions differ, use their average
            B_sq_just_matches = reg_emb * length_mask
            B_sq_just_inserts = tf.reduce_sum(B_sq, axis=1, keepdims=True) - tf.reduce_sum(B_sq_just_matches, axis=1, keepdims=True)
            reg_ins = B_sq_just_inserts / tf.expand_dims(tf.cast(lengths, B.dtype), -1)
        reg = self.L2_match * reg_emb + self.L2_insert * reg_ins
        #zero padding for non match states
        reg *= length_mask
        return reg
        
    def call(self, B, lengths):
        """L2 regularization for each match state.
        Args:
        B: A stack of k emission matrices. Shape: (k, q, s)
        Returns:
        A tensor with the L2 regularization values. Shape: (k, max_model_length)
        """
        #amino acid prior
        l2_loss = self.get_l2_loss(B, lengths)
        return - l2_loss # prior values are assumed to be maximized

    def get_config(self):
        config = super(L2Regularizer, self).get_config()
        config.update({
             "L2_match" : self.L2_match,
             "L2_insert" : self.L2_insert,
             "use_shared_embedding_insertions" : self.use_shared_embedding_insertions
        })
        return config



class JointEmissionPrior(tf.keras.layers.Layer):
    """ Prior over a joint distribution defined by a kernel split. Prior values are added.
        Args: 
            priors: A list of prior objects that will be used for each part of the kernel.
            kernel_split: A list of integers that define the split of the kernel into parts.
                        Should have length one less than the number of priors.
    """
    def __init__(self, priors, kernel_split, **kwargs):
        super(JointEmissionPrior, self).__init__(**kwargs)
        self.priors = priors
        self.kernel_split = kernel_split
        assert len(self.priors) > 2, "Joint emission requires at least 2 priors."
        assert len(self.priors) == len(self.kernel_split) + 1, "Number of priors must be one more than the number of kernel splits."

    def load(self, dtype):
        for prior in self.priors:
            prior.load(dtype)
        
    def call(self, B, lengths):
        """Computes log pdf values for each match state.
        Args:
        B: A stack of k emission matrices. Shape: (k, q, s)
        Returns:
            Shape: (k, q)
        """
        log_joint_pdf = self.priors[0](B[..., :self.kernel_split[0]], lengths)
        for i in range(1, len(self.priors)):
            split_left = self.kernel_split[i-1]
            split_right = self.kernel_split[i] if i < len(self.kernel_split) else None
            log_joint_pdf += self.priors[i](B[..., split_left : split_right], lengths)
        return log_joint_pdf
    
    def get_config(self):
        config = super(JointEmissionPrior, self).get_config()
        config.update({
                    "priors" : self.priors,
                    "kernel_split" : self.kernel_split
                    })
        return config
    
    def __repr__(self):
        return f"JointEmissionPrior(" + ", ".join([str(p) for p in self.priors]) + ")"

    
class NullPrior(tf.keras.layers.Layer):
    """NullObject if no prior should be used.
    """
    def load(self, dtype):
        pass
    
    def call(self, B, lengths):
        k = tf.shape(B)[0]
        max_model_length = tf.reduce_max(lengths)
        return tf.zeros((k, max_model_length), dtype=B.dtype)
    
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
        alpha_global: Favors models with high prob. to enter at the first match and exit after the last match.
        epsilon: A small constant for numerical stability.
    """
    def __init__(self, 
                 match_comp=1, 
                 insert_comp=1, 
                 delete_comp=1,
                 alpha_flank = 7000,
                 alpha_single = 1e9,
                 alpha_global = 1e4,
                 alpha_flank_compl = 1,
                 alpha_single_compl = 1,
                 alpha_global_compl = 1,
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
        self.alpha_global = alpha_global
        self.alpha_flank_compl = alpha_flank_compl
        self.alpha_single_compl = alpha_single_compl
        self.alpha_global_compl = alpha_global_compl
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
        global_prior = []
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
            flank = (self.alpha_flank - 1) * log_probs["unannotated_segment_loop"] #todo: handle as extra case?
            flank += (self.alpha_flank - 1) * log_probs["right_flank_loop"]
            flank += (self.alpha_flank - 1) * log_probs["left_flank_loop"]
            flank += (self.alpha_flank - 1) * log_probs["end_to_right_flank"]
            flank += (self.alpha_flank - 1) * tf.math.log(flank_init_prob[i])
            flank += (self.alpha_flank_compl - 1) * log_probs["unannotated_segment_exit"] #todo: handle as extra case?
            flank += (self.alpha_flank_compl - 1) * log_probs["right_flank_exit"]
            flank += (self.alpha_flank_compl - 1) * log_probs["left_flank_exit"]
            flank += (self.alpha_flank_compl - 1) * tf.math.log(probs["end_to_unannotated_segment"] + probs["end_to_terminal"])
            flank += (self.alpha_flank_compl - 1) * tf.math.log(1-flank_init_prob[i])
            flank_prior.append(tf.squeeze(flank))
            #uni-hit
            hit = (self.alpha_single - 1) * tf.math.log(probs["end_to_right_flank"] + probs["end_to_terminal"]) 
            hit += (self.alpha_single_compl - 1) * tf.math.log(probs["end_to_unannotated_segment"]) 
            hit_prior.append(tf.squeeze(hit))
            #uniform entry/exit prior
            #rescale begin_to_match to sum to 1
            div = tf.math.maximum(self.epsilon, 1- probs["match_to_delete"][0]) 
            btm = probs["begin_to_match"] / div
            enex = tf.expand_dims(btm, 1) * tf.expand_dims(probs["match_to_end"], 0)
            enex = tf.linalg.band_part(enex, 0, -1)
            log_enex = tf.math.log(tf.math.maximum(self.epsilon, 1 - enex))
            log_enex_compl = tf.math.log(tf.math.maximum(self.epsilon, enex))
            glob = (self.alpha_global - 1) * (tf.reduce_sum(log_enex) - log_enex[0, -1])
            glob += (self.alpha_global_compl - 1) * (tf.reduce_sum(log_enex_compl) - log_enex_compl[0, -1])
            global_prior.append( glob )
        prior_val = {
            "match_prior" : match_dirichlet,
            "insert_prior" : insert_dirichlet,
            "delete_prior" : delete_dirichlet,
            "flank_prior" : flank_prior,
            "hit_prior" : hit_prior,
            "global_prior" : global_prior}
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
             "alpha_global" : self.alpha_global, 
             "alpha_flank_compl" : self.alpha_flank_compl, 
             "alpha_single_compl" : self.alpha_single_compl, 
             "alpha_global_compl" : self.alpha_global_compl, 
             "epsilon" : self.epsilon
        }
        config.update(cell_config)
        return config
    
    def __repr__(self):
        return f"ProfileHMMTransitionPrior(match_comp={self.match_comp}, insert_comp={self.insert_comp}, delete_comp={self.delete_comp}, alpha_flank={self.alpha_flank}, alpha_single={self.alpha_single}, alpha_global={self.alpha_global}, alpha_flank_compl={self.alpha_flank_compl}, alpha_single_compl={self.alpha_single_compl}, alpha_global_compl={self.alpha_global_compl})"
    


# class MvnEmbeddingPrior(L2EmbeddingRegularizer):
#     """ A multivariate normal prior for the embedding match states. 
#     """
#     def __init__(self, 
#                  scoring_model_config : Common.ScoringModelConfig,
#                  num_components=Common.PRIOR_DEFAULT_COMPONENTS, 
#                  use_l2=False,
#                  **kwargs):
#         super(MvnEmbeddingPrior, self).__init__(**kwargs)
#         self.scoring_model_config = scoring_model_config
#         self.num_components = num_components
#         self.use_l2 = use_l2


#     def load(self, dtype):
#         super(MvnEmbeddingPrior, self).load(dtype) 
#         prior_path = Common.get_prior_path(self.scoring_model_config, self.num_components)
#         prior_path = os.path.dirname(__file__)+f"/../protein_language_models/"+prior_path
#         print("Loading prior ", prior_path)
#         self.multivariate_normal_prior = make_pdf_model(self.scoring_model_config.dim, 
#                                                         components=self.num_components,
#                                                         precomputed=True, 
#                                                         trainable=False,
#                                                         aggregate_result=False)
#         self.multivariate_normal_prior.load_weights(prior_path)
#         self.multivariate_normal_layer = self.multivariate_normal_prior.layers[5]
#         self.multivariate_normal_layer.trainable = False
#         self.multivariate_normal_layer.compute_values()
        

#     def get_prior_value(self, B_emb, lengths):
#         max_model_length = tf.reduce_max(lengths)
#         length_mask = tf.cast(tf.sequence_mask(lengths), B_emb.dtype)
#         # make sure padding is zero
#         B_emb = B_emb[:,1:max_model_length+1]
#         # compute the prior
#         # reduction and handling sequence length is done internally via the zero padding
#         num_models, num_states, dim  = tf.unstack(tf.shape(B_emb))
#         repeats = tf.cast(dim / self.multivariate_normal_layer.dim, tf.int32)
#         B_emb = tf.reshape(B_emb, (num_models, num_states*repeats, self.multivariate_normal_layer.dim))
#         mvn_log_pdf = self.multivariate_normal_prior(B_emb)
#         mvn_log_pdf = tf.reshape(mvn_log_pdf, (num_models, num_states, repeats))
#         mvn_log_pdf = tf.reduce_sum(mvn_log_pdf, -1)
#         mvn_log_pdf *= length_mask
#         return mvn_log_pdf


#     def call(self, B, lengths):
#         """L2 regularization for each match state.
#         Args:
#         B: A stack of k emission matrices. Shape: (k, q, s)
#         Returns:
#         A tensor with the L2 regularization values. Shape: (k, max_model_length)
#         """
#         #amino acid prior
#         B_amino = B[:,:,:len(SequenceDataset.alphabet)-1]
#         B_emb = B[:,:,len(SequenceDataset.alphabet):len(SequenceDataset.alphabet)+self.scoring_model_config.dim]
#         prior_aa = super().call(B_amino, lengths)
#         prior_emb = self.get_prior_value(B_emb, lengths)
#         if self.use_l2:
#             l2_loss = self.get_l2_loss(B, lengths)
#             return prior_aa + prior_emb - l2_loss #the result is maximized, so we have to negate the regularizer
#         else:
#             return prior_aa + prior_emb


#     def get_config(self):
#         config = super(MvnEmbeddingPrior, self).get_config()
#         config.update({
#              "scoring_model_config" : self.scoring_model_config,
#             "num_components" : self.num_components,
#             "use_l2" : self.use_l2
#         })
#         return config


tf.keras.utils.get_custom_objects()["AminoAcidPrior"] = AminoAcidPrior
tf.keras.utils.get_custom_objects()["L2Regularizer"] = L2Regularizer
tf.keras.utils.get_custom_objects()["JointEmissionPrior"] = JointEmissionPrior
tf.keras.utils.get_custom_objects()["NullPrior"] = NullPrior
tf.keras.utils.get_custom_objects()["ProfileHMMTransitionPrior"] = ProfileHMMTransitionPrior
# tf.keras.utils.get_custom_objects()["L2EmbeddingRegularizer"] = L2EmbeddingRegularizer
# tf.keras.utils.get_custom_objects()["MvnEmbeddingPrior"] = MvnEmbeddingPrior