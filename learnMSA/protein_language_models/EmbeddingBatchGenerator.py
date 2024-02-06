from functools import partial
import os
import gc
import numpy as np
import tensorflow as tf
from learnMSA.msa_hmm.Training import DefaultBatchGenerator, default_model_generator
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset
from learnMSA.protein_language_models.BilinearSymmetric import make_scoring_model
import learnMSA.protein_language_models.Common as Common
from learnMSA.protein_language_models.EmbeddingCache import EmbeddingCache


class EmbeddingBatchGenerator(DefaultBatchGenerator):
    """ Computes batches of input sequences along with static embeddings.
        cache_embeddings: If true, all embeddings will be computed once when configuring the generator and kept in memory. Otherwise they are loaded on the fly.
    """
    def __init__(self, 
                 scoring_model_config : Common.ScoringModelConfig,
                 cache_embeddings=True, 
                 shuffle=True):
        super().__init__(shuffle=shuffle)
        self.scoring_model_config = scoring_model_config
        self.cache_embeddings = cache_embeddings
        self.cache = None


    @tf.function
    def _call_lm_scoring_model(self, lm_inputs, language_model):
        emb = language_model(lm_inputs)
        reduced_emb = self.scoring_layer._reduce(emb, training=False)
        return reduced_emb


    def _compute_reduced_embeddings(self, indices, language_model, encoder):
        seq_batch = [self.data.get_standardized_seq(i) for i in indices]   
        lm_inputs = encoder(seq_batch, np.repeat([[False, False]], len(seq_batch), axis=0))
        return self._call_lm_scoring_model(lm_inputs, language_model)
        

    def configure(self, data : SequenceDataset, config, verbose=False):
        super().configure(data, config)

        # nothing to do if embeddings are already computed
        if self.cache is not None and self.cache.is_filled():
            return

        # load the language model and the scoring model
        # initialize the weights correctly and make sure they are not trainable
        language_model, encoder = Common.get_language_model(self.scoring_model_config.lm_name, 
                                                            max_len = data.max_len+2, trainable=False)
        self.scoring_model = make_scoring_model(self.scoring_model_config, dropout=0.0, trainable=False)    
        scoring_model_path = Common.get_scoring_model_path(self.scoring_model_config)
        self.scoring_model.load_weights(os.path.dirname(__file__)+f"/../protein_language_models/"+scoring_model_path)
        self.scoring_layer = self.scoring_model.layers[-1]
        self.scoring_layer.trainable = False #don't forget to freeze the scoring model!

        if self.cache_embeddings:
            self.cache = EmbeddingCache(self.data.seq_lens, self.scoring_model_config.dim)
            compute_emb_func = partial(self._compute_reduced_embeddings, language_model=language_model, encoder=encoder)
            batch_size_callback = lambda L: max(1, config["batch_size"]([0], L)//2) if callable(config["batch_size"]) else lambda L: max(1, config["batch_size"]//2)
            print("Computing all embeddings (this may take a while).")
            self.cache.fill_cache(compute_emb_func, batch_size_callback, verbose=verbose)
            # once we have cached the embeddings do a cleanup to erase the LM from memory
            tf.keras.backend.clear_session()
            gc.collect()
        else:
            self.language_model, self.encoder = language_model, encoder
                
    def _get_reduced_embedding(self, i):
        """ Returns a 2D tensor of shape (length of sequence i, reduced_embedding_dim) that contains the embeddings of
            the i-th sequence in the dataset used to configure the batch generator.
        """
        if self.cache_embeddings:
            emb = self.cache.get_embedding(i)
        else: #load the embeddings dynamically (not recommended, currently implemented inefficiently)
            emb = self._compute_reduced_embeddings([i], self.language_model, self.encoder)[0]
        return emb
                
    def _pad_and_crop_embeddings(self, embeddings, start, end):
        """ Packs a list of lists of embeddings where each embedding is a 2D tensor into a padded 4D tensor.
            The padding will be zero for all embedding dimensions and one in a new dimension added at the end (the terminal dimension).
            Also crops the sequences accoring to the start and end positions.
        """
        num_models = len(embeddings)
        batch_size = len(embeddings[0])
        max_len = np.amax(end-start)
        dim = embeddings[0][0].shape[1]
        padded_embeddings = np.zeros((num_models, batch_size, max_len+1, dim+1), dtype=np.float32)
        for i,(model_batch,start_batch,end_batch) in enumerate(zip(embeddings, start, end)):
            for j,(emb,s,e) in enumerate(zip(model_batch, start_batch, end_batch)):
                l = e-s
                padded_embeddings[i,j,:l,:-1] = emb[s:e]
                padded_embeddings[i,j,l:,-1] = 1 #terminal dimension
        return padded_embeddings
        
    def __call__(self, indices):
        batch, batch_indices, start, end = super().__call__(indices, return_crop_boundaries=True)
        #retrieve the embeddings for all models and sequences in list-of-lists format
        embeddings = []
        for ind in batch_indices:
            embeddings.append([])
            for i in ind:
                embeddings[-1].append(self._get_reduced_embedding(i))
        #put them in a tensor with padding
        padded_embeddings = self._pad_and_crop_embeddings(embeddings, start, end)
        return batch, batch_indices, padded_embeddings
    
    def get_out_types(self):
        if self.return_only_sequences:
            return (tf.uint8, )
        else:
            return (tf.uint8, tf.int64, tf.float32)  


def make_generic_embedding_model_generator(dim):
    def generic_embedding_model_generator(encoder_layers,
                                        msa_hmm_layer):
        """A generic model generator function. The model inputs are sequences of shape (b, num_model, L) 
            and sequence indices of shape (b, num_model).
        Args:
            dim: The dimension of the embeddings.
            encoder_layers: A list of layers with compatible inputs and outputs and the last output 
                            is compatible with msa_hmm_layer. 
            msa_hmm_layer: An instance of MsaHmmLayer.
        """
        num_models = msa_hmm_layer.cell.num_models
        sequences = tf.keras.Input(shape=(None,None), name="sequences", dtype=tf.uint8)
        indices = tf.keras.Input(shape=(None,), name="indices", dtype=tf.int64)
        embeddings = tf.keras.Input(shape=(None,None,dim+1), name="embeddings", dtype=tf.float32)
        #in the input pipeline, we need the batch dimension to come first to make multi GPU work 
        #we transpose here, because all learnMSA layers require the model dimension to come first
        transposed_sequences = tf.transpose(sequences, [1,0,2])
        transposed_indices = tf.transpose(indices)
        transposed_embeddings = tf.transpose(embeddings, [1,0,2,3])
        forward_seq = transposed_sequences
        for layer in encoder_layers:
            forward_seq = layer(forward_seq, transposed_indices)
        concat_seq = tf.concat([forward_seq, transposed_embeddings], -1)
        loglik = msa_hmm_layer(concat_seq, transposed_indices)
        #transpose back to make model.predict work correctly
        loglik = tf.transpose(loglik)
        model = tf.keras.Model(inputs=[sequences, indices, embeddings], 
                            outputs=[tf.keras.layers.Lambda(lambda x: x, name="loglik")(loglik)])
        return model
    return partial(default_model_generator, generic_gen=generic_embedding_model_generator)