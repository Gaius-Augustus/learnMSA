import tensorflow as tf
import numpy as np
import math
from functools import partial
from learnMSA.msa_hmm.MsaHmmCell import MsaHmmCell
from learnMSA.msa_hmm.MsaHmmLayer import MsaHmmLayer
from learnMSA.msa_hmm.AncProbsLayer import AncProbsLayer
from learnMSA.msa_hmm.Configuration import assert_config
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset
from learnMSA.msa_hmm.Utility import deserialize



class PermuteSeqs(tf.keras.layers.Layer):
    def __init__(self, perm, **kwargs):
        super(PermuteSeqs, self).__init__(**kwargs)
        self.perm = perm

    def call(self, sequences):
        return tf.transpose(sequences, self.perm, name="loglik")

    def get_config(self):
        return {"perm": self.perm}


class Identity(tf.keras.layers.Layer):
    def call(self, x):
        return x


class LearnMSAModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(LearnMSAModel, self).__init__(*args, **kwargs)
        #use mean trackers for scalars to use their update logic
        self.loss_tracker = tf.keras.metrics.Mean()
        self.loglik_tracker = tf.keras.metrics.Mean()
        self.prior_tracker = tf.keras.metrics.Mean()
        self.aux_loss_tracker = tf.keras.metrics.Mean()
        self.use_prior = False

    def loglik(self, y_pred):
        return y_pred[1]

    def prior(self, y_pred):
        return tf.reduce_mean(y_pred[2])

    def aux_loss(self, y_pred):
        return y_pred[3]

    def compute_loss(self, x, y, y_pred, sample_weight):
        if len(y_pred) == 4:
            loss = -self.loglik(y_pred) -self.prior(y_pred) + self.aux_loss(y_pred) 
        else:
            loss = -self.loglik(y_pred)
        loss += sum(self.losses)
        self.loss_tracker.update_state(loss)
        return loss

    def compute_metrics(self, x, y, y_pred, sample_weight):
        metric_results = {"loss": self.loss_tracker.result()}
        self.loglik_tracker.update_state(self.loglik(y_pred))
        metric_results["loglik"] = self.loglik_tracker.result()
        if len(y_pred) == 4:
            self.use_prior = True
            self.prior_tracker.update_state(self.prior(y_pred))
            self.aux_loss_tracker.update_state(self.aux_loss(y_pred))
            metric_results["prior"] = self.prior_tracker.result()
            metric_results["aux_loss"] = self.aux_loss_tracker.result()
        return metric_results

    def reset_metrics(self):
        self.loss_tracker.reset_state()
        self.loglik_tracker.reset_state()
        self.prior_tracker.reset_state()
        self.aux_loss_tracker.reset_state()



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
    transposed_sequences = PermuteSeqs([1,0,2])(sequences)
    transposed_indices = PermuteSeqs([1,0])(indices)
    forward_seq = transposed_sequences
    for layer in encoder_layers:
        forward_seq = layer(forward_seq, transposed_indices)
    if msa_hmm_layer.use_prior:
        loglik, aggregated_loglik, prior, aux_loss = msa_hmm_layer(forward_seq, transposed_indices)
        #transpose back to make model.predict work correctly
        loglik = PermuteSeqs([1,0], name="loglik")(loglik)
        aggregated_loglik = Identity(name="aggregated_loglik")(aggregated_loglik)
        prior = Identity(name="prior")(prior)
        aux_loss = Identity(name="aux_loss")(aux_loss)
        model = LearnMSAModel(inputs=(sequences, indices), 
                            outputs=(loglik, aggregated_loglik, prior, aux_loss))
    else:
        loglik, aggregated_loglik = msa_hmm_layer(forward_seq, transposed_indices)
        #transpose back to make model.predict work correctly
        loglik = PermuteSeqs([1,0], name="loglik")(loglik)
        aggregated_loglik = Identity(name="aggregated_loglik")(aggregated_loglik)
        model = LearnMSAModel(inputs=(sequences, indices), 
                            outputs=(loglik, aggregated_loglik))
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


def make_anc_probs_layer(num_seq, config, clusters=None):
    assert_config(config)
    anc_probs_layer = AncProbsLayer(config["num_models"],
                                    num_seq,
                                    config["num_rate_matrices"],
                                    equilibrium_init=config["encoder_initializer"][2],
                                    rate_init=config["encoder_initializer"][0],
                                    exchangeability_init=config["encoder_initializer"][1],
                                    trainable_rate_matrices=config["trainable_rate_matrices"],
                                    trainable_distances=config["trainable_distances"],
                                    per_matrix_rate=config["per_matrix_rate"],
                                     matrix_rate_init=config["encoder_initializer"][3] if len(config["encoder_initializer"]) > 3 else None,
                                     matrix_rate_l2=config["matrix_rate_l2"],
                                     shared_matrix=config["shared_rate_matrix"],
                                     equilibrium_sample=config["equilibrium_sample"],
                                     transposed=config["transposed"],
                                     clusters=clusters)
    return anc_probs_layer


def default_model_generator(num_seq,
                            effective_num_seq,
                            model_lengths, 
                            config,
                            data : SequenceDataset = None,
                            sequence_weights=None,
                            clusters=None,
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
        clusters: Optional cluster indices for each sequence.
        alphabet_size: Number of symbols without the terminal symbol.
    """
    num_models = config["num_models"]
    assert len(model_lengths) == num_models, \
        (f"The list of given model lengths ({len(model_lengths)}) should"
         + f" match the number of models specified in the configuration({num_models}).")
    msa_hmm_layer = make_msa_hmm_layer(effective_num_seq, model_lengths, config, sequence_weights, alphabet_size)
    anc_probs_layer = make_anc_probs_layer(num_seq, config, clusters)
    model = generic_gen([anc_probs_layer], msa_hmm_layer)
    return model


""" Generates model inputs for training and inference. """
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
        

    def configure(self, data : SequenceDataset, config, cluster_indices=None, verbose=False):
        self.data = data
        self.config = config
        self.cluster_indices = cluster_indices
        self.use_clusters = cluster_indices is not None
        self.num_models = config["num_models"] if "num_models" in config else 1
        self.crop_long_seqs = config["crop_long_seqs"] if "crop_long_seqs" in config else math.inf
        self.permutations = [np.arange(data.num_seq) for _ in range(self.num_models)]
        for p in self.permutations:
            np.random.shuffle(p)
        self.configured = True
        

    """
    Args: indices: A vector of indices in range(0, data.num_seq). If self.shuffle if False, these indices map directly 
                     to the sequences in the dataset. If self.shuffle is True, the indices are permuted for each model.
    Returns:
        A uint8 batch of padded sequences of shape (b, num_models, L) where b is the number of sequences in indices.
        If return_only_sequences is False, the permuted indices and cluster indices are returned as well. 
        The cluster indices will be all zeros if no tree is used. Other wise each leaf/sequence is mapped its parent node.
        If return_crop_boundaries is True, the start and end indices of the cropped sequences are returned.
    """
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

        output = (batch, )
        if not self.return_only_sequences:
            output += (permutated_indices,)
            if self.use_clusters:
                output += (self.cluster_indices[permutated_indices],)
            if return_crop_boundaries:
                output += (start, end)
        return output[0] if len(output) == 1 else output
    

    def get_out_types(self):
        types = (tf.uint8, )
        if not self.return_only_sequences:
            types += (tf.int64, )
        if self.use_clusters:
            types += (tf.int32,)
        return types

    

# batch_generator is a callable object that maps a vector of sequence indices to
# inputs compatible with the model
def make_dataset(indices, batch_generator, batch_size=512, shuffle=True, bucket_by_seq_length=False, model_lengths=[0]): 
    shuffle = shuffle and not bucket_by_seq_length
    batch_generator.shuffle = shuffle
    ds = tf.data.Dataset.from_tensor_slices(indices)
    adaptive_batch = batch_generator.config["batch_size"]
    if bucket_by_seq_length and callable(adaptive_batch): #bucketing only usable if user has not set a fixed batch size
        ds_len = tf.data.Dataset.from_tensor_slices(batch_generator.data.seq_lens[indices].astype(np.int32))
        ds_ind =  tf.data.Dataset.from_tensor_slices(np.arange(indices.size))
        ds = tf.data.Dataset.zip((ds, ds_len, ds_ind))
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
        if bucket_by_seq_length:
            ds_arange = tf.data.Dataset.from_tensor_slices(np.arange(indices.size))
            ds = tf.data.Dataset.zip((ds, ds_arange))
        if shuffle:
            ds = ds.shuffle(indices.size, reshuffle_each_iteration=True)
            ds = ds.repeat()
        ds = ds.batch(batch_size)
        def _batch_func(i):
            batch_data = tf.numpy_function(batch_generator, [i], batch_generator.get_out_types())
            batch, ind = batch_data[:2]
            batch.set_shape(tf.TensorShape([None, batch_generator.num_models, None]))
            ind.set_shape(tf.TensorShape([None, batch_generator.num_models]))
            outputs = (batch, ind)
            if batch_generator.use_clusters:
                clusters = batch_data[2]
                clusters.set_shape(tf.TensorShape([None, batch_generator.num_models]))
                outputs += (clusters,)
            if (batch_generator.use_clusters and len(batch_generator.get_out_types()) == 4 
                or not batch_generator.use_clusters and len(batch_generator.get_out_types()) == 3):
                emb = batch_data[-1]
                emb.set_shape(tf.TensorShape([None, batch_generator.num_models, None, batch_generator.config["scoring_model_config"].dim+1]))
                outputs += (emb,)
            return outputs
        if bucket_by_seq_length:
            def batch_func(i,j):
                return *_batch_func(i), j
        else:
            batch_func = _batch_func
            

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
              clusters=None,
              verbose=True, 
              train_callbacks=[]):
    assert_config(config)
    tf.keras.backend.clear_session() #frees occupied memory 
    tf.get_logger().setLevel('ERROR')
    batch_generator.configure(data, config, verbose=verbose)
    optimizer = tf.keras.optimizers.Adam(config["learning_rate"])
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
                                sequence_weights=sequence_weights,
                                clusters=clusters)
        model.compile(optimizer=optimizer, jit_compile=False)
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
    
    steps = min(max(10, int(100*np.sqrt(indices.shape[0])/batch_size)), 500)
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
    if verbose:
        print("Fitted model successfully.")
    tf.get_logger().setLevel('INFO')
    return model, history

    
tf.keras.utils.get_custom_objects()["PermuteSeqs"] = PermuteSeqs
tf.keras.utils.get_custom_objects()["Identity"] = Identity
tf.keras.utils.get_custom_objects()["LearnMSAModel"] = LearnMSAModel