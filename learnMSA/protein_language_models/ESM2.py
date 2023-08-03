import learnMSA.protein_language_models.Common as common
import tensorflow as tf
from transformers import AutoTokenizer, TFEsmModel, logging
import os


logging.set_verbosity_error()

model_checkpoint = "facebook/esm2_t36_3B_UR50D"
model_checkpoint_s = "facebook/esm2_t33_650M_UR50D" #smaller variant

class ESM2LanguageModel(common.LanguageModel):
    
    def __init__(self, trainable=False, small=False):
        super(ESM2LanguageModel, self).__init__()
        cp = model_checkpoint_s if small else model_checkpoint
        self.model = TFEsmModel.from_pretrained(cp, cache_dir=os.path.dirname(__file__)+"/esm2")
        self.model.trainable = trainable
        self.inputs = self.model.inputs
        self.dim = 1280 if small else 2560
    
    
    def call(self, inputs):
        ids, mask, crop = inputs
        esm2_output = self.model(ids, mask)
        embeddings = tf.cast(esm2_output.last_hidden_state, tf.float32)
        embeddings = self.eliminate_start_stop_tokens(embeddings, crop, mask)
        return embeddings

    def clear_internal_model(self):
        del self.model
        

class ESM2InputEncoder(common.InputEncoder):
    
    def __init__(self, small=False):
        cp = model_checkpoint_s if small else model_checkpoint 
        self.tokenizer = AutoTokenizer.from_pretrained(cp, cache_dir=os.path.dirname(__file__)+"/esm2")
        
    def __call__(self, str_seq, crop):
        tokens = self.tokenizer.batch_encode_plus(str_seq, add_special_tokens=True, padding=True, return_tensors="np")
        ids = tokens["input_ids"]
        mask = tokens["attention_mask"]
        #esm2 uses start- and end-tokens to signalize full proteins
        #cropped sequences should omit the respective token if they were cropped at the start or end
        lens = [len(s) for s in str_seq]
        self.modify_cropped(ids, crop, lens, self.tokenizer.pad_token_id)
        self.modify_cropped(mask, crop, lens, 0)
        return ids, mask, tf.cast(crop, tf.float32)
    
    def get_signature(self):
        return (tf.TensorSpec(shape=(None, None), dtype=tf.int32), 
                 tf.TensorSpec(shape=(None, None), dtype=tf.int32), 
                 tf.TensorSpec(shape=(None, 2), dtype=tf.float32))