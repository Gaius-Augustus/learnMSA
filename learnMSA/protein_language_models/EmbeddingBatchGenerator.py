from functools import partial
import os
import gc
import numpy as np
import tensorflow as tf
from learnMSA.msa_hmm.Training import DefaultBatchGenerator, default_model_generator, PermuteSeqs, Identity, LearnMSAModel
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset
from learnMSA.protein_language_models.BilinearSymmetric import make_scoring_model
import learnMSA.protein_language_models.Common as Common
from learnMSA.protein_language_models.EmbeddingCache import EmbeddingCache



@tf.function
def call_lm_scoring_model(lm_inputs, language_model, scoring_layer):
    emb = language_model(lm_inputs)
    reduced_emb = scoring_layer._reduce(emb, training=False)
    return reduced_emb


""" Computes embeddings from a language model and a scoring model.
Args:
    indices: A list of sequence indices.
    data: A SequenceDataset object.
    language_model: A language model that takes a batch of sequences and returns embeddings.
    encoder: An input encoder that converts sequences to the input format of the language model.
    scoring_layer: The scoring layer of the scoring model that reduces the embeddings to the desired dimension.
"""
def compute_reduced_embeddings(indices, 
                               data : SequenceDataset, 
                               language_model : Common.LanguageModel, 
                               encoder : Common.InputEncoder, 
                               scoring_layer):
    seq_batch = [data.get_standardized_seq(i) for i in indices]   
    lm_inputs = encoder(seq_batch, np.repeat([[False, False]], len(seq_batch), axis=0))
    return call_lm_scoring_model(lm_inputs, language_model, scoring_layer)


""" A generic embedding function with cached models.
"""
def generic_compute_embedding_func(data : SequenceDataset, scoring_model_config : Common.ScoringModelConfig):
    # load the language model and the scoring model
    # initialize the weights correctly and make sure they are not trainable
    language_model, encoder = Common.get_language_model(scoring_model_config.lm_name, 
                                                        max_len = data.max_len+2, trainable=False)
    scoring_model = make_scoring_model(scoring_model_config, dropout=0.0, trainable=False)    
    scoring_model_path = Common.get_scoring_model_path(scoring_model_config)
    scoring_model.load_weights(os.path.dirname(__file__)+f"/../protein_language_models/"+scoring_model_path)
    scoring_layer = scoring_model.layers[-1]
    scoring_layer.trainable = False #don't forget to freeze the scoring model!
    return partial(compute_reduced_embeddings, data=data, language_model=language_model, encoder=encoder, scoring_layer=scoring_layer)
    



class EmbeddingBatchGenerator(DefaultBatchGenerator):
    """ Computes batches of input sequences along with static embeddings.
    Args:
        compute_embedding_func: A function that computes the embeddings for a batch of sequence indices.
        cache_embeddings: If true, all embeddings will be computed once when configuring the generator and kept in memory. Otherwise they are loaded on the fly.
    """
    def __init__(self, 
                 compute_embedding_func,
                 cache_embeddings=True, 
                 shuffle=True):
        super().__init__(shuffle=shuffle)
        self.compute_embedding_func = compute_embedding_func
        self.cache_embeddings = cache_embeddings
        self.cache = None
        
    """
    Args:
        <as DefaultBatchGenerator>
    """
    def configure(self, data : SequenceDataset, config, cluster_indices=None, verbose=False):
        super().configure(data, config, cluster_indices, verbose)

        # nothing to do if embeddings are already computed
        if self.cache is not None and self.cache.is_filled():
            return

        if self.cache_embeddings:
            self.cache = EmbeddingCache(self.data.seq_lens, config["scoring_model_config"].dim)
            batch_size_callback = (lambda L: max(1, config["batch_size"]([0], L)//2)) if callable(config["batch_size"]) else (lambda L: max(1, config["batch_size"]//2))
            print("Computing all embeddings (this may take a while).")
            self.cache.fill_cache(self.compute_embedding_func, batch_size_callback, verbose=verbose)
            # once we have cached the embeddings do a cleanup to erase the LM from memory
            del self.compute_embedding_func #remove cached LM
            tf.keras.backend.clear_session()
            gc.collect()

                
    def _get_reduced_embedding(self, i):
        """ Returns a 2D tensor of shape (length of sequence i, reduced_embedding_dim) that contains the embeddings of
            the i-th sequence in the dataset used to configure the batch generator.
        """
        if self.cache_embeddings:
            emb = self.cache.get_embedding(i)
        else: #load the embeddings dynamically (not recommended, currently implemented inefficiently)
            emb = self._compute_reduced_embeddings([i])[0]
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
        if self.use_clusters:
            batch, batch_indices, clusters, start, end = super().__call__(indices, return_crop_boundaries=True)
        else:
            batch, batch_indices, start, end = super().__call__(indices, return_crop_boundaries=True)
        #retrieve the embeddings for all models and sequences in list-of-lists format
        embeddings = []
        for ind in batch_indices:
            embeddings.append([])
            for i in ind:
                embeddings[-1].append(self._get_reduced_embedding(i))
        #put them in a tensor with padding
        padded_embeddings = self._pad_and_crop_embeddings(embeddings, start, end)
        if self.use_clusters:
            return batch, batch_indices, clusters, padded_embeddings
        else:
            return batch, batch_indices, padded_embeddings
    
    def get_out_types(self):
        types = (tf.uint8, )
        if not self.return_only_sequences:
            types += (tf.int64,)
        if self.use_clusters:
            types += (tf.int32,)
        if not self.return_only_sequences:
            types += (tf.float32,)
        return types

    def sample_embedding_variance(self, n_samples=10000):
        """ Approximates the variance of embedding dimensions. """
        if self.cache_embeddings:
            if self.cache is None or not self.cache.is_filled():
                raise ValueError("Cannot sample embedding variance before embeddings are cached. Call configure().")
            return np.var(self.cache.cache, axis=0)
        else:
            raise NotImplementedError("Sampling embedding variance is not supported when embeddings are not cached.")


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
        transposed_sequences = PermuteSeqs([1,0,2])(sequences)
        transposed_indices = PermuteSeqs([1,0])(indices)
        transposed_embeddings = PermuteSeqs([1,0,2,3])(embeddings)
        forward_seq = transposed_sequences
        for layer in encoder_layers:
            forward_seq = layer(forward_seq, transposed_indices)
        concat_seq = tf.keras.layers.Concatenate()([forward_seq, transposed_embeddings])
        loglik, aggregated_loglik, prior, aux_loss = msa_hmm_layer(concat_seq, transposed_indices)
        #transpose back to make model.predict work correctly
        loglik = PermuteSeqs([1,0], name="loglik")(loglik)
        aggregated_loglik = Identity(name="aggregated_loglik")(aggregated_loglik)
        prior = Identity(name="prior")(prior)
        aux_loss = Identity(name="aux_loss")(aux_loss)
        model = LearnMSAModel(inputs=(sequences, indices, embeddings), 
                        outputs=(loglik, aggregated_loglik, prior, aux_loss))
        return model
    return partial(default_model_generator, generic_gen=generic_embedding_model_generator)