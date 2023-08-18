import learnMSA.protein_language_models.Common as common
import tensorflow as tf
from transformers import TFT5EncoderModel, T5Tokenizer, logging
import numpy as np
import re
import os


logging.set_verbosity_error()

class ProtT5LanguageModel(common.LanguageModel):
    
    def __init__(self, trainable=False):
        super(ProtT5LanguageModel, self).__init__()
        self.model = TFT5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", from_pt=True, cache_dir=os.path.dirname(__file__)+"/protT5_model")
        self.model.trainable = trainable
        self.inputs = self.model.inputs
        self.dim = 1024
        
    def call(self, inputs):
        ids, mask = inputs[0], inputs[1]
        protT5_output = self.model(ids, mask)
        embeddings = tf.cast(protT5_output.last_hidden_state[:,:-1], tf.float32)
        mask = mask[:,1:] #mask also contains one special token per sequence, do not count it
        max_len = tf.reduce_max(tf.reduce_sum(mask, -1))
        mask = tf.cast(tf.expand_dims(mask, -1), tf.float32)
        embeddings = (embeddings * mask)[:,:max_len]
        return embeddings
    
    def clear_internal_model(self):
        del self.model
        


class ProtT5InputEncoder(common.InputEncoder):
    
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False, cache_dir=os.path.dirname(__file__)+"/protT5_model")
    
    def __call__(self, str_seq, crop):
        #add whitespaces between residues and replace uncommon amino acids with X
        str_seq = [re.sub(r"[UZOB]", "X", " ".join(sequence)) for sequence in str_seq]
        ids = self.tokenizer.batch_encode_plus(str_seq, add_special_tokens=True, padding=True, return_tensors="tf")
        #protT5 uses a relative position embedding
        #we don't have to indicate cropped sequences to the model
        return ids['input_ids'], ids['attention_mask']
    
    def get_signature(self):
        return (tf.TensorSpec(shape=(None, None), dtype=tf.int32), 
                 tf.TensorSpec(shape=(None, None), dtype=tf.int32))