import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
from learnMSA.msa_hmm.MsaHmmCell import MsaHmmCell
from learnMSA.msa_hmm.MsaHmmLayer import MsaHmmLayer
from learnMSA.msa_hmm.AncProbsLayer import AncProbsLayer
from learnMSA.msa_hmm.Configuration import assert_config

        

def generic_model_generator(encoder_layers,
                            msa_hmm_layer):
    """A generic model generator function. The model inputs are sequences of shape (num_model, b, L) 
        and sequence indices of shape (num_model, b).
    Args:
        encoder_layers: A list of layers with compatible inputs and outputs and the last output 
                        is compatible with msa_hmm_layer. 
        msa_hmm_layer: An instance of MsaHmmLayer.
    """
    num_models = msa_hmm_layer.cell.num_models
    sequences = tf.keras.Input(shape=(None,None,), name="sequences", dtype=tf.uint8)
    indices = tf.keras.Input(shape=(None,), name="indices", dtype=tf.int64)
    forward_seq = sequences
    for layer in encoder_layers:
        forward_seq = layer(forward_seq, indices)
    loglik = msa_hmm_layer(forward_seq)
    model = tf.keras.Model(inputs=[sequences, indices], 
                        outputs=[tf.keras.layers.Lambda(lambda x: x, name="loglik")(loglik)])
    return model

def make_msa_hmm_layer(effective_num_seq,
                        model_lengths, 
                        config,
                        alphabet_size=25):
    """Constructs a cell and a MSA HMM layer given a config.
    """
    assert_config(config)
    msa_hmm_cell = MsaHmmCell(model_lengths, config["emitter"], config["transitioner"])
    msa_hmm_layer = MsaHmmLayer(msa_hmm_cell, 
                                effective_num_seq,
                                use_prior=config["use_prior"],
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
                            alphabet_size=25):
    """A callback that constructs the default learnMSA model.
    Args:
        num_seq: The total number of sequences to align.
        effective_num_seq: The actual number of sequences currently used for training (model surgery might use only a subset).
        model_lengths: List of pHMM lengths.
        config: Dictionary storing the configuration.
        alphabet_size: Number of symbols without the terminal symbol (i.e. 25 for amino acids).
    """
    num_models = config["num_models"]
    assert len(model_lengths) == num_models, \
        (f"The list of given model lengths ({len(model_lengths)}) should"
         + f" match the number of models specified in the configuration({num_models}).")
    msa_hmm_layer = make_msa_hmm_layer(effective_num_seq, model_lengths, config, alphabet_size)
    anc_probs_layer = make_anc_probs_layer(num_seq, config)
    model = generic_model_generator([anc_probs_layer], msa_hmm_layer)
    return model


class DefaultBatchGenerator():
    def __init__(self, fasta_file, num_models, return_only_sequences=False, shuffle=True, alphabet_size=25):
        self.fasta_file = fasta_file
        #generate a unique permutation of the sequence indices for each model to train
        self.num_models = num_models
        self.permutations = [np.arange(fasta_file.num_seq) for _ in range(num_models)]
        for p in self.permutations:
            np.random.shuffle(p)
        self.return_only_sequences = return_only_sequences
        self.alphabet_size = alphabet_size
        self.shuffle = shuffle
        
    def __call__(self, indices):
        #use a different permutation of the sequences per trained model
        if self.shuffle:
            permutated_indices = np.stack([perm[indices] for perm in self.permutations], axis=0)
        else:
            permutated_indices = np.stack([indices]*self.num_models, axis=0)
        max_len = np.max(self.fasta_file.seq_lens[permutated_indices])
        batch = np.zeros((self.num_models, indices.shape[0], max_len+1), dtype=np.uint8) 
        batch += self.alphabet_size #initialize with terminal symbols
        for k,perm_ind in enumerate(permutated_indices):
            for i,j in enumerate(perm_ind):
                batch[k, i, :self.fasta_file.seq_lens[j]] = self.fasta_file.get_raw_seq(j)
        if self.return_only_sequences:
            return batch
        else:
            return batch, permutated_indices
    
    def get_out_types(self):
        if self.return_only_sequences:
            return (tf.uint8)
        else:
            return (tf.uint8, tf.int64)  
    
          
# batch_generator is a callable object that maps a vector of sequence indices to
# inputs compatible with the model
def make_dataset(indices, batch_generator, batch_size=512, shuffle=True):   
    batch_generator.shuffle = shuffle
    ds = tf.data.Dataset.from_tensor_slices(indices)
    if shuffle:
        ds = ds.shuffle(indices.size, reshuffle_each_iteration=True)
        ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.map(lambda i: tf.numpy_function(func=batch_generator,
                inp=[i], Tout=batch_generator.get_out_types()),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=True)
    ds_y = tf.data.Dataset.from_tensor_slices(tf.zeros(1)).batch(batch_size).repeat()
    ds = tf.data.Dataset.zip((ds, ds_y))
    ds = ds.prefetch(tf.data.AUTOTUNE) #preprocessings and training steps in parallel
    return ds
    

def fit_model(model_generator,
              batch_generator,
              fasta_file,
              indices,
              model_lengths, 
              config,
              batch_size, 
              epochs,
              verbose=True):
    assert_config(config)
    tf.keras.backend.clear_session() #frees occupied memory 
    tf.get_logger().setLevel('ERROR')
    optimizer = tf.optimizers.Adam(config["learning_rate"])
    if verbose:
        print("Fitting models of lengths", model_lengths, "on", indices.shape[0], "sequences.")
        print("Batch size=", batch_size, "Learning rate=", config["learning_rate"])
    def make_and_compile():
        model = model_generator(num_seq=fasta_file.num_seq,
                                effective_num_seq=indices.shape[0],
                                model_lengths=model_lengths,
                                config=config)
        model.compile(optimizer=optimizer)
        return model
    num_gpu = len([x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']) 
    if verbose:
        print("Using", num_gpu, "GPUs.")
    if num_gpu > 1:       
        
        #workaround: https://github.com/tensorflow/tensorflow/issues/50487
        import atexit
        mirrored_strategy = tf.distribute.MirroredStrategy()    
        atexit.register(mirrored_strategy._extended._collective_ops._pool.close) # type: ignore
        
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




