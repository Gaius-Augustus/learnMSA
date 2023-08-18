import learnMSA.protein_language_models.Common as common
import tensorflow as tf
import os
from proteinbert import load_pretrained_model
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
from proteinbert.existing_model_loading import DEFAULT_REMOTE_MODEL_DUMP_URL
from proteinbert.tokenization import additional_token_to_index



class ProteinBERTLanguageModel(common.LanguageModel):
    def __init__(self, model, trainable=False):
        super(ProteinBERTLanguageModel, self).__init__()
        self.model = model
        self.model.trainable = trainable
        self.inputs = model.inputs
        self.dim = 1562
        
    def call(self, inputs):
        proteinbert_output = self.model(inputs[:2])
        crop = inputs[2]
        #get rid of global annotations
        proteinbert_seq_input, embeddings = inputs[0], proteinbert_output[0] 
        #mask start-, end- and paddings markers
        mask = tf.cast(proteinbert_seq_input < 25, embeddings.dtype)
        embeddings = self.eliminate_start_stop_tokens(embeddings, crop, mask)
        return embeddings
    
    def clear_internal_model(self):
        del self.model
    

class ProteinBERTInputEncoder(common.InputEncoder):
    def __init__(self, input_encoder, max_len):
        self.input_encoder = input_encoder
        self.max_len = max_len
        
    def __call__(self, str_seq, crop):
        seq, glob = self.input_encoder.encode_X(str_seq, self.max_len)
        #proteinBERT uses start- and end-tokens to signalize full proteins
        #cropped sequences should omit the respective token if they were cropped at the start or end
        self.modify_cropped(seq, crop, [len(s) for s in str_seq], additional_token_to_index["<PAD>"])
        return seq, tf.cast(glob, tf.float32), tf.cast(crop, tf.float32)
    
    def get_signature(self):
        return (tf.TensorSpec(shape=(None, None), dtype=tf.int32), 
                 tf.TensorSpec(shape=(None, None), dtype=tf.float32), 
                 tf.TensorSpec(shape=(None, 2), dtype=tf.float32))
    

def get_proteinBERT_model_and_encoder(max_len, trainable=False, model_dir=os.path.dirname(__file__)+"/proteinBERT_model"):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    pretrained_model_generator, input_encoder = load_pretrained_model(local_model_dump_dir = model_dir,
                                                                      local_model_dump_file_name = os.path.basename(DEFAULT_REMOTE_MODEL_DUMP_URL),
                                                                      download_model_dump_if_not_exists=True, 
                                                                      validate_downloading = False)
    proteinbert_model = pretrained_model_generator.create_model(max_len, compile=False)
    proteinbert_model = get_model_with_hidden_layers_as_outputs(proteinbert_model)
    return ProteinBERTLanguageModel(proteinbert_model, trainable), ProteinBERTInputEncoder(input_encoder, max_len)