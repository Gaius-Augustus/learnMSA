import tensorflow as tf
import numpy as np



PRIOR_PATH = "priors_V3"
SCORING_MODEL_PATH = "new_scoring_models_frozen"
PRIOR_DEFAULT_COMPONENTS = 100

class ScoringModelConfig():
    def __init__(self, 
                 lm_name="esm2",
                 dim=64,
                 activation="softmax",
                 use_aa=False,
                 scaled=True,
                 suffix=""):
        self.lm_name = lm_name
        self.dim = dim
        self.activation = activation
        self.use_aa = use_aa
        self.scaled = scaled
        self.suffix = suffix

    def __repr__(self):
        return f"ScoringModelConfig(lm_name={self.lm_name}, dim={self.dim}, activation={self.activation}, suffix={self.suffix})"



def get_scoring_model_path(config : ScoringModelConfig):
    return f"{SCORING_MODEL_PATH}/{config.lm_name}_{config.dim}_{config.activation}{config.suffix}/checkpoints"


def get_prior_path(config : ScoringModelConfig, components):
    return f"{PRIOR_PATH}/{config.lm_name}_{config.dim}_reduced_mix{components}_{config.activation}{config.suffix}/checkpoints"


## Constructs and loads a language model with contextual imports.
def get_language_model(name, max_len=512, trainable=False):
    if name == "proteinBERT":
        import learnMSA.protein_language_models.ProteinBERT as ProteinBERT
        language_model, encoder = ProteinBERT.get_proteinBERT_model_and_encoder(max_len = max_len+2, trainable=trainable)
    elif name == "esm2":
        import learnMSA.protein_language_models.ESM2 as ESM2
        language_model, encoder = ESM2.ESM2LanguageModel(trainable=trainable), ESM2.ESM2InputEncoder()
    elif name == "esm2s":
        import learnMSA.protein_language_models.ESM2 as ESM2
        language_model, encoder = ESM2.ESM2LanguageModel(trainable=trainable, small=True), ESM2.ESM2InputEncoder(small=True)
    elif name == "protT5":
        import learnMSA.protein_language_models.ProtT5 as ProtT5
        language_model, encoder = ProtT5.ProtT5LanguageModel(trainable=trainable), ProtT5.ProtT5InputEncoder()
    else:
        raise ValueError(f"Language model {name} not supported.")
    return language_model, encoder


class LanguageModel(tf.keras.layers.Layer):
    """ Base class for language models that generate residual-level embeddings of the input sequences.
    """
    def call(self, inputs):
        pass
    
    def eliminate_start_stop_tokens(self, embeddings, crop, mask):
        mask = tf.cast(mask, embeddings.dtype)
        mask_crop_1 = tf.concat([mask[:, 1:], tf.zeros_like(mask[:, :1])], 1)
        mask_crop_2 = tf.concat([mask[:, 2:], tf.zeros_like(mask[:, :2])], 1)
        # both tokens
        mask_no_start_stop = mask_crop_2 * (1 - crop[:, :1]) * (1 - crop[:, 1:])
        # only start token
        mask_no_start_stop += mask_crop_1 * crop[:, :1] * (1 - crop[:, 1:])
        # only end token
        mask_no_start_stop += mask_crop_1 * (1 - crop[:, :1]) * crop[:, 1:]
        # no start- or end-token
        mask_no_start_stop += mask * crop[:, :1] * crop[:, 1:]
        # shift sequences with a start token by 1 
        embeddings_no_start = tf.concat([embeddings[:,1:], tf.zeros_like(embeddings[:,:1])], 1)
        embeddings_no_start_stop = embeddings_no_start * crop[:, :1, tf.newaxis] + embeddings * (1 - crop[:, :1, tf.newaxis])
        embeddings_no_start_stop *= mask_no_start_stop[:,:,tf.newaxis]
        # crop all padding-only columns
        max_len = tf.reduce_max(tf.reduce_sum(tf.cast(mask_no_start_stop, tf.int32), -1))
        embeddings_no_start_stop = embeddings_no_start_stop[:,:max_len] 
        return embeddings_no_start_stop 
        
        


class InputEncoder():
    """ Base class for encoders that map proteins as strings to input tensors compatible with the specific language model.
        The output of the encoder should be the input to the corresponding language model.
    """
    def __call__(self, str_seq, crop):
        pass
    
    def get_signature(self):
        pass
    
    def modify_cropped(self, x, crop, lens, pad_id):
        for i,(cs,ce) in enumerate(crop):
            if cs:
                x[i] = np.roll(x[i], -1)
                x[i, -1] = pad_id
                if ce:
                    x[i, lens[i]] = pad_id
            elif ce:
                x[i, lens[i]+1] = pad_id
                
    
    
# for convenience
dims = {
    "proteinBERT" : 1562,
    "esm2" : 2560,
    "protT5" : 1024
}