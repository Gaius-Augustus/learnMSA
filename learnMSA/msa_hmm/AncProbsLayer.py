import tensorflow as tf
import numpy as np
import learnMSA.msa_hmm.Utility as ut
import learnMSA.msa_hmm.Fasta as fasta


TAU_MAX = 2.5
     
    
class TauConstraint(tf.keras.constraints.Constraint):
    
    def __init__(self):
        super(TauConstraint, self).__init__()
        self.non_neg = tf.keras.constraints.NonNeg()

    def __call__(self, w):
        return tf.cast(-self.non_neg(TAU_MAX - self.non_neg(tf.cast(w, tf.float32))) + TAU_MAX, ut.dtype)
    
    
    
class AncProbsLayer(tf.keras.layers.Layer): 
    def __init__(self, num_seqs, tau_init=0.0):
        super(AncProbsLayer, self).__init__()
        self.Q = np.eye(fasta.s, dtype=np.float32)
        self.Q[:20, :20],_,_ = ut.read_paml_file() #extended transition rate matrix
        self.num_seqs = num_seqs
        self.tau_init = tau_init
        self.tau_constraint = TauConstraint()
    
        
    def build(self, input_shape):
        self.tau = self.add_weight(shape=[self.num_seqs], 
                                   name="tau", 
                                   initializer=tf.constant_initializer(self.tau_init),
                                   dtype=ut.dtype,
                                  trainable=True)
        
        
    def get_tau(self, subset=None):
        if subset is None:
            return self.tau_constraint(self.tau)
        else:
            return self.tau_constraint(tf.gather(self.tau, subset))

        
    #expects mask to indicate all std. amino acid positions, i.e. argmax(pos) < 20
    #subset indicates which  sequences of the complete set are included in the batch
    def call(self, inputs, mask, subset):
        inputs = tf.cast(inputs, ut.dtype)
        mask = tf.cast(mask, ut.dtype)
        tau_subset = self.get_tau(subset)
        part_std = ut.make_anc_probs(inputs, tf.cast(self.Q, ut.dtype), tau_subset) * mask
        part_other = inputs * (1-mask)
        return part_std + part_other