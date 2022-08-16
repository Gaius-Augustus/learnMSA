import tensorflow as tf
import numpy as np
import learnMSA.msa_hmm.Utility as ut


class TauConstraint(tf.keras.constraints.Constraint):
    
    def __init__(self, tau_max=2.5):
        super(TauConstraint, self).__init__()
        self.non_neg = tf.keras.constraints.NonNeg()
        self.tau_max = tau_max

    def __call__(self, w):
        w = -self.non_neg(self.tau_max - self.non_neg(w)) + self.tau_max
        return w
    
    
    
class AncProbsLayer(tf.keras.layers.Layer): 
    def __init__(self, 
                 num_seqs, 
                 tau_init=tf.keras.initializers.Zeros(),
                 name="AncProbsLayer",
                 **kwargs):
        super(AncProbsLayer, self).__init__(name=name, **kwargs)
        self.Q = np.eye(25+1, dtype=np.float32) #amino acid alphabet including terminal symbol for convenience
        self.Q[:20, :20],_,_ = ut.read_paml_file() #extended transition rate matrix
        self.num_seqs = num_seqs
        self.tau_init = tau_init
        self.tau_constraint = TauConstraint()
    
        
    def build(self, input_shape):
        self.tau = self.add_weight(shape=[self.num_seqs], 
                                   name="tau", 
                                   initializer=self.tau_init,
                                   trainable=True)
        
        
    def get_tau(self, subset=None):
        if subset is None:
            return self.tau_constraint(self.tau)
        else:
            return self.tau_constraint(tf.gather(self.tau, subset))

        
    #subset indicates which  sequences of the complete set are included in the batch
    def call(self, inputs, subset):
        std_aa_mask = tf.reduce_sum(inputs[:,:,:20], -1, keepdims=True)
        tau_subset = self.get_tau(subset)
        part_std = ut.make_anc_probs(inputs, self.Q, tau_subset) * std_aa_mask
        part_other = inputs * (1-std_aa_mask)
        return part_std + part_other