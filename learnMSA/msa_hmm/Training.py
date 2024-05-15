import tensorflow as tf
import numpy as np
import math
from functools import partial
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
    msa_hmm_cell = MsaHmmCell(model_lengths, 
                                dim = 24 * config["num_rate_matrices"],
                                emitter = config["emitter"], 
                                transitioner = config["transitioner"])
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
        
    def configure(self, data : SequenceDataset, config, verbose=False):
        self.data = data
        self.config = config
        self.num_models = config["num_models"] if "num_models" in config else 1
        self.crop_long_seqs = config["crop_long_seqs"] if "crop_long_seqs" in config else math.inf
        self.permutations = [np.arange(data.num_seq) for _ in range(self.num_models)]
        for p in self.permutations:
            np.random.shuffle(p)
        self.configured = True
        
    def __call__(self, indices, return_crop_boundaries=False):
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
        if return_crop_boundaries:
            start = np.zeros((indices.shape[0], self.num_models), dtype=np.int32)
            end = np.zeros((indices.shape[0], self.num_models), dtype=np.int32)
        batch += self.alphabet_size #initialize with terminal symbols
        for i,perm_ind in enumerate(permutated_indices):
            for k,j in enumerate(perm_ind):
                if return_crop_boundaries:
                    seq, start[i, k], end[i, k] = self.data.get_encoded_seq(j, crop_to_length=self.crop_long_seqs, return_crop_boundaries=True)
                else:
                    seq = self.data.get_encoded_seq(j, crop_to_length=self.crop_long_seqs, return_crop_boundaries=False)
                batch[i, k, :min(self.data.seq_lens[j], self.crop_long_seqs)] = seq
        if self.return_only_sequences:
            if return_crop_boundaries:
                return batch, start, end
            else: 
                return batch
        else:
            if return_crop_boundaries:
                return batch, permutated_indices, start, end 
            else: 
                return batch, permutated_indices
    
    def get_out_types(self):
        if self.return_only_sequences:
            return (tf.uint8, )
        else:
            return (tf.uint8, tf.int64) 
        
    
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
    ds_y = tf.data.Dataset.from_tensor_slices(tf.zeros(1)).batch(batch_size).repeat()
    ds = tf.data.Dataset.zip((ds, ds_y))
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
              verbose=True, 
              train_callbacks=[]):
    assert_config(config)
    tf.keras.backend.clear_session() #frees occupied memory 
    tf.get_logger().setLevel('ERROR')
    batch_generator.configure(data, config, verbose)
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
    callbacks = [termiante_on_nan, early_stopping] + train_callbacks

    if config["use_language_model"]:
        msa_hmm_layer = None
        for i, layer in enumerate(model.layers[1:]):
            if layer.name.startswith("msa_hmm_layer"):
                msa_hmm_layer = layer
                break
        assert msa_hmm_layer is not None, "Can not find a MsaHmmLayer in the specified model."

        # class CustomCallback(tf.keras.callbacks.Callback):

        #     # def on_train_begin(self, logs=None):
        #     #     msa_hmm_layer.cell.emitter[0].step_counter.assign(0.)
        #     #     msa_hmm_layer.reverse_cell.emitter[0].step_counter.assign(0.)

        #     def on_train_batch_end(self, batch, logs=None):
        #         msa_hmm_layer.cell.emitter[0].step_counter.assign_add(1.)
        #         msa_hmm_layer.reverse_cell.emitter[0].step_counter.assign_add(1.)
        # callbacks.append(CustomCallback())

    history = model.fit(dataset, 
                        epochs=epochs,
                        steps_per_epoch=steps,
                          callbacks=callbacks,
                        verbose = 2*int(verbose))
    tf.get_logger().setLevel('INFO')
    return model, history