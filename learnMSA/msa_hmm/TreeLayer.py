import tensorflow as tf

# tmp include
sys.path.insert(0, "../TensorTree")
import tensortree 

tensortree.set_backend("tensorflow")


""" A layer over a fixed tree topology that does two things:
    1. It computes the ancestral probabilities of the sequences.
    2. It adds a loss that maximizes the likelihood 
"""
class TreeLayer(tf.keras.layers.Layer): 
    pass