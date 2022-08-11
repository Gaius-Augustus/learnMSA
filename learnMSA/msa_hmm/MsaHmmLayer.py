import os
import tensorflow as tf
import numpy as np
from tensorflow.python.client import device_lib
from learnMSA.msa_hmm.MsaHmmCell import MsaHmmCell
import learnMSA.msa_hmm.Utility as ut
import learnMSA.msa_hmm.DirichletMixture as dm

class MsaHmmLayer(tf.keras.layers.Layer):
    def __init__(self, 
                 length, 
                 num_seq,
                 emission_init,
                 transition_init, 
                 flank_init,
                 alpha_flank,
                 alpha_single,
                 alpha_frag ,
                 trainable_kernels={},
                 use_prior=True,
                 dirichlet_mix_comp_count=1
                ):
        super(MsaHmmLayer, self).__init__()
        self.length = length
        self.num_seq = num_seq
        self.use_prior = use_prior 
        self.load_priors(alpha_flank, alpha_single, alpha_frag, dirichlet_mix_comp_count)
        if isinstance(emission_init, str) and emission_init == "background":
            self.emission_init = make_emission_init_kernel(length)
        else:
            self.emission_init = emission_init
        if isinstance(transition_init, str) and transition_init in transition_init_kernels:
            self.transition_init = make_transition_init_kernel(length, **transition_init_kernels[transition_init])
        elif isinstance(transition_init, str) and transition_init == "infer_from_prior" and self.use_prior:
            self.transition_init = self.make_transition_init_from_prior()
        else:
            self.transition_init = transition_init
        if isinstance(flank_init, str):
            self.flank_init = flank_init_values[flank_init]
        else:
            self.flank_init = flank_init
        self.trainable_kernels = trainable_kernels

        
    def build(self, input_shape):
        self.C = MsaHmmCell(self.length, 
                            self.emission_init,
                            self.transition_init,
                            self.flank_init,
                            self.trainable_kernels)
        self.F = tf.keras.layers.RNN(self.C, return_sequences=True, return_state=True)
        
        
    def call(self, inputs, training=False):
        self.C.init_cell()
        inputs = tf.cast(inputs, ut.dtype)
        _, _, loglik = self.F(inputs, 
                                    initial_state=self.C.get_initial_state(batch_size=tf.shape(inputs)[0]))
        loglik_mean = tf.reduce_mean(loglik) 
        if self.use_prior:
            prior = self.get_prior_log_density()
            prior /= self.num_seq
            MAP = loglik_mean + prior
            self.add_loss(tf.squeeze(-MAP))
        else:
            self.add_loss(tf.squeeze(-loglik_mean))
        if training:
            self.add_metric(loglik_mean, "loglik")
            if self.use_prior:
                self.add_metric(prior, "logprior")
        return loglik
    
    
    def load_priors(self, 
                    alpha_flank, 
                    alpha_single, 
                    alpha_frag,
                    dirichlet_mix_comp_count):
        #euqivalent to the alpha parameters of a dirichlet mixture -1 .
        #these values are crutial when jointly optimizing the main model with the additional
        #"Plan7" states and transitions
        #favors high probability of staying one of the 3 flaking states
        self.alpha_flank = tf.constant(alpha_flank, dtype=ut.dtype) 
        #favors high probability for a single main model hit (avoid looping paths)
        self.alpha_single = tf.constant(alpha_single, dtype=ut.dtype) 
        #favors models with high prob. to enter at the first match and exit after the last match
        self.alpha_frag = tf.constant(alpha_frag, dtype=ut.dtype) 
        
        PRIOR_PATH = os.path.dirname(__file__)+"/trained_prior/"
        model, DMP = dm.make_model(dirichlet_mix_comp_count, 20, -1, trainable=False)
        model.load_weights(
            PRIOR_PATH+str(dirichlet_mix_comp_count)+"_components_prior_pdf/ckpt").expect_partial()
        self.emission_dirichlet_mix = dm.DirichletMixturePrior(dirichlet_mix_comp_count, 20, -1,
                                            DMP.alpha_kernel.numpy(),
                                            DMP.q_kernel.numpy(),
                                            trainable=False)
        model, DMP = dm.make_model(1, 20, -1, trainable=False)
        model.load_weights(
            PRIOR_PATH+"1_components_prior_pdf/ckpt").expect_partial()
        self.emission_dirichlet_1 = dm.DirichletMixturePrior(1, 20, -1,
                                            DMP.alpha_kernel.numpy(),
                                            DMP.q_kernel.numpy(),
                                            trainable=False)
        model, DMP = dm.make_model(1, 3, -1, trainable=False)
        model.load_weights(
            PRIOR_PATH+"transition_priors/match/ckpt").expect_partial()
        self.match_dirichlet = dm.DirichletMixturePrior(1, 3, -1,
                                            DMP.alpha_kernel.numpy(),
                                            DMP.q_kernel.numpy(),
                                            trainable=False)
        model, DMP = dm.make_model(1, 2, -1, trainable=False)
        model.load_weights(
            PRIOR_PATH+"transition_priors/insert/ckpt").expect_partial()
        self.insert_dirichlet = dm.DirichletMixturePrior(1, 2, -1,
                                            DMP.alpha_kernel.numpy(),
                                            DMP.q_kernel.numpy(),
                                            trainable=False)
        model, DMP = dm.make_model(1, 2, -1, trainable=False)
        model.load_weights(
            PRIOR_PATH+"transition_priors/delete/ckpt").expect_partial()
        self.delete_dirichlet = dm.DirichletMixturePrior(1, 2, -1,
                                            DMP.alpha_kernel.numpy(),
                                            DMP.q_kernel.numpy(),
                                            trainable=False) 
        
    
    def get_prior_log_density(self):      
        probs = self.C.make_probs()
        log_probs = {part_name : tf.math.log(p) for part_name, p in probs.items()}
        prior = 0
        #match states
        p_match = tf.stack([probs["match_to_match"],
                            probs["match_to_insert"],
                            probs["match_to_delete"][1:]], axis=-1)
        p_match_sum = tf.reduce_sum(p_match, axis=-1, keepdims=True)

        prior += tf.reduce_sum(self.match_dirichlet.log_pdf(p_match / p_match_sum))
        #insert states
        p_insert = tf.stack([probs["insert_to_match"],
                           probs["insert_to_insert"]], axis=-1)
        prior += tf.reduce_sum(self.insert_dirichlet.log_pdf(p_insert))
        #delete states
        p_delete = tf.stack([probs["delete_to_match"][:-1],
                           probs["delete_to_delete"]], axis=-1)
        prior += tf.reduce_sum(self.delete_dirichlet.log_pdf(p_delete))

        #match state emissions
        B = self.C.make_B()[1:self.C.length+1,:20]
        B /= tf.reduce_sum(B, axis=-1, keepdims=True)
        dmp_prior = tf.reduce_sum(self.emission_dirichlet_mix.log_pdf(B))
        prior += dmp_prior 

        prior += self.alpha_flank * log_probs["unannotated_segment_loop"]
        prior += self.alpha_flank * log_probs["right_flank_loop"]
        prior += self.alpha_flank * log_probs["left_flank_loop"]
        prior += self.alpha_flank * tf.math.log(probs["end_to_right_flank"])
        prior += self.alpha_flank * tf.math.log(tf.math.sigmoid(self.C.flank_init))

        prior += self.alpha_single * tf.math.log(probs["end_to_right_flank"] + probs["end_to_terminal"])
        
        #uniform entry/exit prior
        enex = tf.expand_dims(probs["begin_to_match"], 1) * tf.expand_dims(probs["match_to_end"], 0)
        enex = tf.linalg.band_part(enex, 0, -1)
        enex = tf.math.log(1 - enex) 
        prior += self.alpha_frag * (tf.reduce_sum(enex) - enex[0, -1])
        return prior
    
    
    def make_transition_init_from_prior(self):
        flank_prob = (self.alpha_flank+1)/(self.alpha_flank+2)
        no_repeat_prob = (self.alpha_single+1)/(self.alpha_single+2)
        match_dist = np.log(self.match_dirichlet.expectation())
        insert_dist = np.log(self.insert_dirichlet.expectation())
        delete_dist = np.log(self.delete_dirichlet.expectation())
        values = transition_init_kernels["default"].copy()
        values["MM"] = match_dist[0]
        values["MI"] = match_dist[1]
        values["MD"] = match_dist[2]
        values["DM"] = delete_dist[0]
        values["DD"] = delete_dist[1]
        values["IM"] = insert_dist[0]
        values["II"] = insert_dist[1]
        #values["FC"] = np.log(flank_prob)
        #values["FE"] = np.log(1-flank_prob)
        #values["R"] = np.log(1-no_repeat_prob)
        #values["RF"] = np.log(no_repeat_prob/2)
        #values["T"] = np.log(no_repeat_prob/2)
        return make_transition_init_kernel(self.length, **values)
    
    
    
def make_emission_init_kernel(length):
    return np.stack([np.log(ut.background_distribution)]*length)


def make_transition_init_kernel(length, MM, MI, MD, 
                           II, IM, 
                           DM, DD,
                           FC, FE,
                           R, RF, T,
                           EN1=None, EN=None, EX=None):
    if EN1 is None or EN is None or EX is None:
        entry =  np.zeros(length)
        #choose such that entry[0] will always be ~0.5 independent of model length
        entry[1:] += np.log(1/(length-1)) 
        exit =  np.zeros(length)
        #choose such that all exit probs equal the probs entry[i] for i > 0 
        p_exit_desired = 0.5 / (length-1)
        match_3 = tf.nn.softmax(np.array([MM, MI, MD], dtype=float))*(1-p_exit_desired)
        MM = np.log(match_3[0])
        MI = np.log(match_3[1])
        MD = np.log(match_3[2])
        exit[:-1] += np.log(p_exit_desired)
    else:
        entry = np.zeros(length)+EN
        entry[0] = EN1
        exit = np.zeros(length)+EX
    transition_init_kernel = {
        "begin_to_match" : entry,
        "match_to_end" : exit,
        "match_to_match" : np.zeros(length-1)+MM,
        "match_to_insert" : np.zeros(length-1)+MI,
        "insert_to_match" : np.zeros(length-1)+IM,
        "insert_to_insert" : np.zeros(length-1)+II,
        "match_to_delete" : np.zeros(length)+MD,
        "delete_to_match" : np.zeros(length)+DM,
        "delete_to_delete" : np.zeros(length-1)+DD,
        "left_flank_loop" : np.array([FC]),
        "left_flank_exit" : np.array([FE]),
        "right_flank_loop" : np.array([FC]),
        "right_flank_exit" : np.array([FE]),
        "unannotated_segment_loop" : np.array([FC]),
        "unannotated_segment_exit" : np.array([FE]),
        "end_to_unannotated_segment" : np.array([R]),
        "end_to_right_flank" : np.array([RF]),
        "end_to_terminal" : np.array([T]) }
    return transition_init_kernel
        
    
transition_init_kernels = {
     "default" :  {"MM" : 1, 
                    "MI" : -1,
                    "MD" : -1,
                    "II" : -0.5,
                    "IM" : 0,
                    "DM" : 0,
                    "DD" : -0.5,
                    "FC" : 0,
                    "FE" : -1,
                    "R" : -9,
                    "RF" : 0, 
                    "T" : 0}}

flank_init_values = {
    "default" : 0.0
}