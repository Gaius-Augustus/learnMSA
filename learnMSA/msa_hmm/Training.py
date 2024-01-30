import tensorflow as tf
import numpy as np
import gc
import os
import math
from functools import partial
from learnMSA.protein_language_models.BilinearSymmetric import make_scoring_model
from learnMSA.msa_hmm.MsaHmmCell import MsaHmmCell
from learnMSA.msa_hmm.MsaHmmLayer import MsaHmmLayer
from learnMSA.msa_hmm.AncProbsLayer import AncProbsLayer
from learnMSA.msa_hmm.Configuration import assert_config
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset

        

def generic_model_generator(encoder_layers,
                            msa_hmm_layer):
    """A generic model generator function. The model inputs are sequences of shape (b, num_model, L) 
        and sequence indices of shape (b, num_model).
    Args:
        encoder_layers: A list of layers with compatible inputs and outputs and the last output 
                        is compatible with msa_hmm_layer. 
        msa_hmm_layer: An instance of MsaHmmLayer.
    """
    num_models = msa_hmm_layer.cell.num_models
    sequences = tf.keras.Input(shape=(None,None), name="sequences", dtype=tf.uint8)
    indices = tf.keras.Input(shape=(None,), name="indices", dtype=tf.int64)
    #in the input pipeline, we need the batch dimension to come first to make multi GPU work 
    #we transpose here, because all learnMSA layers require the model dimension to come first
    transposed_sequences = tf.transpose(sequences, [1,0,2])
    transposed_indices = tf.transpose(indices)
    forward_seq = transposed_sequences
    for layer in encoder_layers:
        forward_seq = layer(forward_seq, transposed_indices)
    loglik = msa_hmm_layer(forward_seq, transposed_indices)
    #transpose back to make model.predict work correctly
    loglik = tf.transpose(loglik)
    model = tf.keras.Model(inputs=[sequences, indices], 
                        outputs=[tf.keras.layers.Lambda(lambda x: x, name="loglik")(loglik)])
    return model

def make_msa_hmm_layer(effective_num_seq,
                        model_lengths, 
                        config,
                        sequence_weights=None,
                        alphabet_size=len(SequenceDataset.alphabet)-1):
    """Constructs a cell and a MSA HMM layer given a config.
    """
    assert_config(config)
    msa_hmm_cell = MsaHmmCell(model_lengths, config["emitter"], config["transitioner"])
    msa_hmm_layer = MsaHmmLayer(msa_hmm_cell, 
                                effective_num_seq,
                                use_prior=config["use_prior"],
                                sequence_weights=sequence_weights,
                                dtype=tf.float32)
    return msa_hmm_layer

def make_anc_probs_layer(num_seq, config):
    assert_config(config)
    anc_probs_layer = AncProbsLayer(config["num_models"],
                                    num_seq,
                                    config["num_rate_matrices"],
                                    equilibrium_init=config["encoder_initializer"][2],
                                    rate_init=config["encoder_initializer"][0],
                                    exchangeability_init=config["encoder_initializer"][1],
                                    trainable_rate_matrices=config["trainable_rate_matrices"],
                                    per_matrix_rate=config["per_matrix_rate"],
                                     matrix_rate_init=config["encoder_initializer"][3] if len(config["encoder_initializer"]) > 3 else None,
                                     matrix_rate_l2=config["matrix_rate_l2"],
                                     shared_matrix=config["shared_rate_matrix"],
                                     equilibrium_sample=config["equilibrium_sample"],
                                     transposed=config["transposed"])
    return anc_probs_layer

def default_model_generator(num_seq,
                            effective_num_seq,
                            model_lengths, 
                            config,
                            data : SequenceDataset = None,
                            sequence_weights=None,
                            alphabet_size=len(SequenceDataset.alphabet)-1,
                            generic_gen=generic_model_generator):
    """A callback that constructs the default learnMSA model. Can be used as a template for custom generators.
    Args:
        num_seq: The total number of sequences to align.
        effective_num_seq: The actual number of sequences currently used for training (model surgery might use only a subset).
        model_lengths: List of pHMM lengths.
        config: Dictionary storing the configuration.
        data: The sequence dataset corresponding to this model. Typically but not necessarily equal to the training data of the model. 
              This is part of the model generator callback because the generated model might depends on some sequence information like maximum length.
        sequence_weights: Optional likelihood weights per sequence.
        alphabet_size: Number of symbols without the terminal symbol.
    """
    num_models = config["num_models"]
    assert len(model_lengths) == num_models, \
        (f"The list of given model lengths ({len(model_lengths)}) should"
         + f" match the number of models specified in the configuration({num_models}).")
    msa_hmm_layer = make_msa_hmm_layer(effective_num_seq, model_lengths, config, sequence_weights, alphabet_size)
    anc_probs_layer = make_anc_probs_layer(num_seq, config)
    model = generic_gen([anc_probs_layer], msa_hmm_layer)
    return model



class DefaultBatchGenerator():
    def __init__(self, 
                return_only_sequences=False, 
                shuffle=True, 
                alphabet_size=len(SequenceDataset.alphabet)-1):
        #generate a unique permutation of the sequence indices for each model to train
        self.return_only_sequences = return_only_sequences
        self.alphabet_size = alphabet_size
        self.shuffle = shuffle
        self.configured = False
        
    def configure(self, data : SequenceDataset, config):
        self.data = data
        self.config = config
        self.num_models = config["num_models"] if "num_models" in config else 1
        self.crop_long_seqs = config["crop_long_seqs"] if "crop_long_seqs" in config else math.inf
        self.permutations = [np.arange(data.num_seq) for _ in range(self.num_models)]
        for p in self.permutations:
            np.random.shuffle(p)
        self.configured = True
        
    def __call__(self, indices):
        if not self.configured:
            raise ValueError("A batch generator must be configured with the configure(data, config) method.") 
        #use a different permutation of the sequences per trained model
        if self.shuffle:
            permutated_indices = np.stack([perm[indices] for perm in self.permutations], axis=1)
        else:
            permutated_indices = np.stack([indices]*self.num_models, axis=1)
        max_len = np.max(self.data.seq_lens[permutated_indices])
        max_len = min(max_len, self.crop_long_seqs)
        batch = np.zeros((indices.shape[0], self.num_models, max_len+1), dtype=np.uint8) 
        batch += self.alphabet_size #initialize with terminal symbols
        for i,perm_ind in enumerate(permutated_indices):
            for k,j in enumerate(perm_ind):
                batch[i, k, :min(self.data.seq_lens[j], self.crop_long_seqs)] = self.data.get_encoded_seq(j, crop_to_length=self.crop_long_seqs)
        if self.return_only_sequences:
            return batch
        else:
            return batch, permutated_indices
    
    def get_out_types(self):
        if self.return_only_sequences:
            return (tf.uint8, )
        else:
            return (tf.uint8, tf.int64) 
        
        
        
class EmbeddingBatchGenerator(DefaultBatchGenerator):
    # only import contextual when lm features are required

    """ Computes batches of input sequences along with static embeddings.
        cache_embeddings: If true, all embeddings will be computed once when configuring the generator and kept in memory. Otherwise they are loaded on the fly.
    """
    def __init__(self, 
                 lm_name,
                 reduced_embedding_dim=32,
                 cache_embeddings=True, 
                 use_finetuned_lm=True,
                 shuffle=True):
        super().__init__(shuffle=shuffle)
        self.lm_name = lm_name
        self.reduced_embedding_dim = reduced_embedding_dim
        self.cache_embeddings = cache_embeddings
        self.use_finetuned_lm = use_finetuned_lm
        if self.cache_embeddings:
            self.embedding_cache = []
            
    def _load_language_model(self, data : SequenceDataset):
        if self.lm_name == "proteinBERT":
            from learnMSA.protein_language_models import ProteinBERT
            language_model, encoder = ProteinBERT.get_proteinBERT_model_and_encoder(max_len = data.max_len+2)
        elif self.lm_name == "esm2":
            from learnMSA.protein_language_models import ESM2
            language_model, encoder = ESM2.ESM2LanguageModel(), ESM2.ESM2InputEncoder()
        elif self.lm_name == "protT5":
            from learnMSA.protein_language_models import ProtT5
            language_model, encoder = ProtT5.ProtT5LanguageModel(), ProtT5.ProtT5InputEncoder()
        if self.use_finetuned_lm:
            language_model.model.load_weights(os.path.dirname(__file__)+f"finetuned_models/{self.lm_name}_{self.reduced_embedding_dim}/checkpoints")
        return language_model, encoder
    
    def _compute_reduced_embeddings(self, seqs, language_model, encoder):
        lm_inputs = encoder(seqs, np.repeat([[False, False]], len(seqs), axis=0))
        emb = language_model(lm_inputs)
        bilinear_symmetric_layer = self.scoring_model.layers[-1]
        reduced_emb = bilinear_symmetric_layer._reduce(emb, training=False, softmax_on_reduce_dim=False)
        return reduced_emb
        
    def configure(self, data : SequenceDataset, config):
        super().configure(data, config)
        language_model, encoder = self._load_language_model(data)
        self.scoring_model = make_scoring_model(language_model.dim, self.reduced_embedding_dim, dropout=0.0)
        if self.use_finetuned_lm:
            self.scoring_model.load_weights(os.path.dirname(__file__)+f"/../protein_language_models/scoring_models/{self.lm_name}_{self.reduced_embedding_dim}/checkpoints")
        else:
            self.scoring_model.load_weights(os.path.dirname(__file__)+f"/../protein_language_models/scoring_models_frozen/{self.lm_name}_{self.reduced_embedding_dim}/checkpoints")
        self.scoring_model.layers[-1].trainable = False #don't forget to freeze the scoring model!
        if self.cache_embeddings:
            if len(self.embedding_cache) == 0:
                self.batch_size = config["batch_size"]([0], data.max_len) if callable(config["batch_size"]) else config["batch_size"]
                if self.lm_name == "esm2s":
                    self.batch_size //= 2
                if self.lm_name == "protT5":
                    self.batch_size //= 4
                if self.lm_name == "esm2":
                    self.batch_size //= 4
                print("Computing all embeddings (this may take a while).")
                for i in range(0, data.num_seq, self.batch_size):
                    seq_batch = [data.get_standardized_seq(j) for j in range(i, min(i+self.batch_size, data.num_seq))]      
                    emb = self._compute_reduced_embeddings(seq_batch, language_model, encoder).numpy() #move to cpu 
                    for j in range(emb.shape[0]):
                        self.embedding_cache.append(emb[j, :data.seq_lens[i+j]])
                # once we have cached the (lower dimensional) embeddings do a cleanup
                tf.keras.backend.clear_session()
                gc.collect()
        else:
            self.language_model, self.encoder = language_model, encoder
                
    def _get_reduced_embedding(self, i):
        """ Returns a 2D tensor of shape (length of sequence i, reduced_embedding_dim) that contains the embeddings of
            the i-th sequence in the dataset used to configure the batch generator.
        """
        if self.cache_embeddings:
            emb = self.embedding_cache[i]
        else: #load the embeddings dynamically (not recommended, currently implemented inefficiently)
            seq = self.data.get_standardized_seq(i)
            emb = self._compute_reduced_embeddings([seq], self.language_model, self.encoder)[0]
        return emb
                
    def _pad_embeddings(self, embeddings):
        """ Packs a list of lists of embeddings where each embedding is a 2D tensor into a padded 4D tensor.
            The padding will be zero for all embedding dimensions and one in a new dimension added at the end (the terminal dimension).
        """
        num_models = len(embeddings)
        batch_size = len(embeddings[0])
        max_len = max([emb.shape[0] for model_batch in embeddings for emb in model_batch])
        dim = embeddings[0][0].shape[1]
        padded_embeddings = np.zeros((num_models, batch_size, max_len+1, dim+1), dtype=np.float32)
        for i,model_batch in enumerate(embeddings):
            for j,emb in enumerate(model_batch):
                l = emb.shape[0]
                padded_embeddings[i,j,:l,:-1] = emb
                padded_embeddings[i,j,l:,-1] = 1 #terminal dimension
        return padded_embeddings
        
    def __call__(self, indices):
        batch, batch_indices = super().__call__(indices)
        #retrieve the embeddings for all models and sequences in list-of-lists format
        embeddings = []
        for ind in batch_indices:
            embeddings.append([])
            for i in ind:
                embeddings[-1].append(self._get_reduced_embedding(i))
        #put them in a tensor with padding
        padded_embeddings = self._pad_embeddings(embeddings)
        return batch, batch_indices, padded_embeddings
    
    def get_out_types(self):
        if self.return_only_sequences:
            return (tf.uint8, )
        else:
            return (tf.uint8, tf.int64, tf.float32)  
    
          
# batch_generator is a callable object that maps a vector of sequence indices to
# inputs compatible with the model
def make_dataset(indices, batch_generator, batch_size=512, shuffle=True, bucket_by_seq_length=False, model_lengths=[0]):   
    shuffle = shuffle and not bucket_by_seq_length
    batch_generator.shuffle = shuffle
    ds = tf.data.Dataset.from_tensor_slices(indices)
    if bucket_by_seq_length:
        ds_len = tf.data.Dataset.from_tensor_slices(batch_generator.data.seq_lens[indices].astype(np.int32))
        ds_ind =  tf.data.Dataset.from_tensor_slices(np.arange(indices.size))
        ds = tf.data.Dataset.zip((ds, ds_len, ds_ind))
        adaptive_batch = batch_generator.config["batch_size"]
        if not callable(adaptive_batch):
            raise ValueError("""Batch generator must be configured with a configuration that support adaptive batch size callsback,
                                if bucket_by_seq_length is True.""")
        bucket_boundaries = [200, 520, 700, 850, 1200, 2000, 4000, math.inf]
        bucket_batch_sizes = [adaptive_batch(model_lengths, b) for b in bucket_boundaries]
        ds = ds.bucket_by_sequence_length(
                                element_length_func=lambda i,L,j: L,
                                bucket_boundaries=bucket_boundaries[:-1],
                                bucket_batch_sizes=bucket_batch_sizes)

        batch_func_out_types = batch_generator.get_out_types() + (tf.int64,)
        func = (lambda i,j: (batch_generator(i), j)) if len(batch_func_out_types) == 2 else lambda i,j: (*batch_generator(i), j)
        batch_func = lambda i,_,j: tf.numpy_function(func=func, inp=[i,j], Tout=batch_func_out_types)
    else:
        if shuffle:
            ds = ds.shuffle(indices.size, reshuffle_each_iteration=True)
            ds = ds.repeat()
        ds = ds.batch(batch_size)

        batch_func = lambda i: tf.numpy_function(func=batch_generator, inp=[i], Tout=batch_generator.get_out_types())

    ds = ds.map(batch_func,
                # no parallel processing if using an indexed dataset
                num_parallel_calls=None if batch_generator.data.indexed else tf.data.AUTOTUNE,
                deterministic=True)
    if not batch_generator.data.indexed:
        ds = ds.prefetch(2) #preprocessings and training steps in parallel
    # get rid of a warning, see https://github.com/tensorflow/tensorflow/issues/42146
    # in case of multi GPU, we want to split the data dimension accross GPUs
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    ds = ds.with_options(options)
    ds = tf.data.Dataset.zip((ds, tf.data.Dataset.from_tensor_slices(tf.zeros(1)).repeat()))
    return ds
    

def fit_model(model_generator,
              batch_generator,
              data : SequenceDataset,
              indices,
              model_lengths, 
              config,
              batch_size, 
              epochs,
              sequence_weights=None,
              verbose=True):
    assert_config(config)
    tf.keras.backend.clear_session() #frees occupied memory 
    tf.get_logger().setLevel('ERROR')
    batch_generator.configure(data, config)
    optimizer = tf.optimizers.Adam(config["learning_rate"])
    if verbose:
        print("Fitting models of lengths", model_lengths, "on", indices.shape[0], "sequences.")
        print("Batch size=", batch_size, "Learning rate=", config["learning_rate"])
        if sequence_weights is not None:
            print("Using sequence weights ", sequence_weights, ".")
        else:
            print("Don't use sequence weights.")
        if batch_generator.crop_long_seqs < math.inf:
            num_cropped = np.sum(data.seq_lens[indices] > batch_generator.crop_long_seqs)
            if num_cropped > 0:
                print(f"""{num_cropped} sequences are longer than {batch_generator.crop_long_seqs} and will be cropped for training.""")
                print("To disable cropping, use --crop disable. To change the cropping limit to X, use --crop X.")
    def make_and_compile():
        model = model_generator(num_seq=data.num_seq,
                                effective_num_seq=indices.shape[0],
                                model_lengths=model_lengths,
                                config=config,
                                data=data,
                                sequence_weights=sequence_weights)
        model.compile(optimizer=optimizer)
        return model
    num_gpu = len([x.name for x in tf.config.list_logical_devices() if x.device_type == 'GPU']) 
    if verbose:
        print("Using", num_gpu, "GPUs.")
    if num_gpu > 1:       
        if config["use_language_model"]:
            print("Found multiple GPUs, but using a language model is currently not supported in multi-GPU mode. Using single GPU.")
            model = make_and_compile()
        else:
            #workaround: https://github.com/tensorflow/tensorflow/issues/50487
            #import atexit
            mirrored_strategy = tf.distribute.MirroredStrategy()    
            #atexit.register(mirrored_strategy._extended._collective_ops._pool.close) # type: ignore
            
            with mirrored_strategy.scope():
                model = make_and_compile()
    else:
         model = make_and_compile()
    steps = max(10, int(100*np.sqrt(indices.shape[0])/batch_size))
    dataset = make_dataset(indices, 
                           batch_generator, 
                           batch_size,
                           shuffle=True)
    termiante_on_nan = tf.keras.callbacks.TerminateOnNaN()
    early_stopping = tf.keras.callbacks.EarlyStopping("loss", patience=1)
    callbacks = [termiante_on_nan, early_stopping]
    history = model.fit(dataset, 
                        epochs=epochs,
                        steps_per_epoch=steps,
                          callbacks=callbacks,
                        verbose = 2*int(verbose))
    tf.get_logger().setLevel('INFO')
    return model, history


def generic_embedding_model_generator(encoder_layers,
                                      msa_hmm_layer):
    """A generic model generator function. The model inputs are sequences of shape (b, num_model, L) 
        and sequence indices of shape (b, num_model).
    Args:
        encoder_layers: A list of layers with compatible inputs and outputs and the last output 
                        is compatible with msa_hmm_layer. 
        msa_hmm_layer: An instance of MsaHmmLayer.
    """
    num_models = msa_hmm_layer.cell.num_models
    sequences = tf.keras.Input(shape=(None,None), name="sequences", dtype=tf.uint8)
    indices = tf.keras.Input(shape=(None,), name="indices", dtype=tf.int64)
    embeddings = tf.keras.Input(shape=(None,None,33), name="embeddings", dtype=tf.float32)
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


embedding_model_generator = partial(default_model_generator, generic_gen=generic_embedding_model_generator)