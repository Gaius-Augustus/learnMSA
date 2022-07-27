import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
from msa_hmm.MsaHmmCell import MsaHmmCell
from msa_hmm.MsaHmmLayer import MsaHmmLayer
from msa_hmm.AncProbsLayer import AncProbsLayer
import msa_hmm.Utility as ut



def default_model_generator(num_seq,
                            effective_num_seq,
                            model_length, 
                            config,
                            alphabet_size=25):
    sequences = tf.keras.Input(shape=(None, alphabet_size+1), name="sequences", dtype=tf.float32)
    subset = tf.keras.Input(shape=(), name="subset", dtype=tf.int32)
    
    msa_hmm_cell = MsaHmmCell(model_length,    
                              input_dim = alphabet_size, 
                              emission_init = config["emission_init"],
                              transition_init = config["transition_init"],
                              insertion_init = config["insertion_init"],
                              flank_init = config["flank_init"],
                              alpha_flank = config["alpha_flank"],
                              alpha_single = config["alpha_single"],
                              alpha_frag = config["alpha_frag"],
                              frozen_insertions = config["frozen_insertions"])
    msa_hmm_layer = MsaHmmLayer(msa_hmm_cell, effective_num_seq)
    anc_probs_layer = AncProbsLayer(num_seq, config["encoder_initializer"][0])
    
    if config["use_anc_probs"]:
        forward_seq = anc_probs_layer(sequences, subset)
    else:
        forward_seq = sequences
    loglik = msa_hmm_layer(forward_seq)
    
    model = tf.keras.Model(inputs=[sequences, subset], 
                        outputs=[tf.keras.layers.Lambda(lambda x: x, name="loglik")(loglik)])
    return model
                            
    
class DefaultBatchGenerator():
    def __init__(self, fasta_file, alphabet_size=25):
        self.fasta_file = fasta_file
        self.alphabet_size = alphabet_size
        
    def __call__(self, indices):
        max_len = np.max(self.fasta_file.seq_lens[indices])
        batch = np.zeros((indices.shape[0], max_len+1)) + self.alphabet_size
        for i,j in enumerate(indices):
            batch[i, :self.fasta_file.seq_lens[j]] = self.fasta_file.get_raw_seq(j)
        batch = tf.one_hot(batch, self.alphabet_size+1)
        return batch, indices
    
    def get_out_types(self):
        return (tf.float32, tf.int64)  
    
    
class OnlySequencesBatchGenerator():
    def __init__(self, fasta_file, alphabet_size=25):
        self.fasta_file = fasta_file
        self.alphabet_size = alphabet_size
        
    def __call__(self, indices):
        max_len = np.max(self.fasta_file.seq_lens[indices])
        batch = np.zeros((indices.shape[0], max_len+1)) + self.alphabet_size
        for i,j in enumerate(indices):
            batch[i, :self.fasta_file.seq_lens[j]] = self.fasta_file.get_raw_seq(j)
        batch = tf.one_hot(batch, self.alphabet_size+1)
        return batch
    
    def get_out_types(self):
        return (tf.float32)  
        
          
# batch_generator is a callable object that maps a vector of sequence indices to
# inputs compatible with the model
def make_dataset(indices, batch_generator, batch_size=512, shuffle=True):   
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
              model_length, 
              config,
              batch_size, 
              epochs,
              verbose=True):
    tf.keras.backend.clear_session() #frees occupied memory 
    tf.get_logger().setLevel('ERROR')
    optimizer = tf.optimizers.Adam(config["learning_rate"])
    if verbose:
        print("Fitting a model of length", model_length, "on", indices.shape[0], "sequences.")
        print("Batch size=", batch_size, "Learning rate=", config["learning_rate"])
    def make_and_compile():
        model = model_generator(num_seq=fasta_file.num_seq,
                                effective_num_seq=indices.shape[0],
                                model_length=model_length,
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
    steps = max(30, int(250*np.sqrt(fasta_file.num_seq)/batch_size))
    dataset = make_dataset(indices, 
                           batch_generator, 
                           batch_size,
                           shuffle=True)
    history = model.fit(dataset, 
                        epochs=epochs,
                        steps_per_epoch=steps,
                        verbose = 1*int(verbose))
    tf.get_logger().setLevel('INFO')
    return model, history




