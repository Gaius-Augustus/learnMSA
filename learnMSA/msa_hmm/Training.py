import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
from learnMSA.msa_hmm.MsaHmmLayer import MsaHmmLayer
from learnMSA.msa_hmm.AncProbsLayer import AncProbsLayer
import learnMSA.msa_hmm.Utility as ut
import learnMSA.msa_hmm.Fasta as fasta



def make_model(num_seq,
               effective_num_seq,
               model_length, 
               emission_init,
               transition_init,
               flank_init,
               alpha_flank, 
               alpha_single, 
               alpha_frag,
               use_prior=True,
               dirichlet_mix_comp_count=1,
               use_anc_probs=True,
               tau_init=0.0, 
               trainable_kernels={}):
    sequences = tf.keras.Input(shape=(None,fasta.s), name="sequences", dtype=ut.dtype)
    mask = tf.keras.Input(shape=(None,1), name="mask", dtype=ut.dtype)
    subset = tf.keras.Input(shape=(), name="subset", dtype=tf.int32)
    msa_hmm_layer = MsaHmmLayer(length=model_length,
                                num_seq=effective_num_seq,
                                emission_init=emission_init,
                                transition_init=transition_init,
                                flank_init=flank_init,
                                alpha_flank=alpha_flank, 
                                alpha_single=alpha_single, 
                                alpha_frag=alpha_frag,
                                trainable_kernels=trainable_kernels,
                                use_prior=use_prior,
                                dirichlet_mix_comp_count=dirichlet_mix_comp_count)
    anc_probs_layer = AncProbsLayer(num_seq, tau_init=tau_init)
    if use_anc_probs:
        forward_seq = anc_probs_layer(sequences, mask, subset)
    else:
        forward_seq = sequences
    loglik = msa_hmm_layer(forward_seq)
    model = tf.keras.Model(inputs=[sequences, mask, subset], 
                        outputs=[tf.keras.layers.Lambda(lambda x: x, name="loglik")(loglik)])
    return model, msa_hmm_layer, anc_probs_layer  
    
    
    
def make_dataset(fasta_file, batch_size, shuffle=True, indices=None):
    if indices is None:
        indices = tf.range(fasta_file.num_seq)
    def get_seq(i):
        seq = fasta_file.get_raw_seq(i)
        seq = np.append(seq, [fasta.s-1]) #terminal symbol
        seq = seq.astype(np.int32)
        return (seq, tf.cast(i, tf.int64))
    def preprocess_seq(seq, i):
        std_aa_mask = tf.expand_dims(seq < 20, -1)
        std_aa_mask = tf.cast(std_aa_mask, dtype=ut.dtype)
        return tf.one_hot(seq, fasta.s, dtype=ut.dtype), std_aa_mask, i
    ds = tf.data.Dataset.from_tensor_slices(indices)
    if shuffle:
        ds = ds.shuffle(fasta_file.num_seq, reshuffle_each_iteration=True)
        ds = ds.repeat()
    ds = ds.map(lambda i: tf.numpy_function(func=get_seq,
                inp=[i], Tout=(tf.int32, tf.int64)),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=True)
    ds = ds.padded_batch(batch_size, 
                         padded_shapes=([None], []),
                         padding_values=(tf.constant(fasta.s-1, dtype=tf.int32), 
                                         tf.constant(0, dtype=tf.int64)))
    ds = ds.map(preprocess_seq)
    ds_y = tf.data.Dataset.from_tensor_slices(tf.zeros(1)).batch(batch_size).repeat()
    ds = tf.data.Dataset.zip((ds, ds_y))
    ds = ds.prefetch(tf.data.AUTOTUNE) #preprocessings and training steps in parallel
    return ds
    

    
def fit_model(fasta_file, 
              indices, 
              model_length, 
              emission_init,
              transition_init,
              flank_init,
              alpha_flank, 
              alpha_single, 
              alpha_frag,
              use_prior=True,
              dirichlet_mix_comp_count=1,
              use_anc_probs=True,
              tau_init=0.0, 
              trainable_kernels={},
              batch_size=256, 
              learning_rate=0.1,
              epochs=4,
              verbose=True):
    tf.keras.backend.clear_session() #frees occupied memory 
    tf.get_logger().setLevel('ERROR')
    optimizer = tf.optimizers.Adam(learning_rate)
    if verbose:
        print("Fitting a model of length", model_length, "on", indices.shape[0], "sequences")
        print("Batch size=",batch_size, "Learning rate=",learning_rate)
    def make_and_compile():
        model, msa_hmm_layer, anc_probs_layer = make_model(num_seq=fasta_file.num_seq,
                                                           effective_num_seq=indices.shape[0],
                                                           model_length=model_length, 
                                                           emission_init=emission_init,
                                                           transition_init=transition_init,
                                                           flank_init=flank_init,
                                                           alpha_flank=alpha_flank, 
                                                           alpha_single=alpha_single, 
                                                           alpha_frag=alpha_frag,
                                                           use_prior=use_prior,
                                                           dirichlet_mix_comp_count=dirichlet_mix_comp_count,
                                                           use_anc_probs=use_anc_probs,
                                                           tau_init=tau_init, 
                                                           trainable_kernels=trainable_kernels)
        model.compile(optimizer=optimizer)
        return model, msa_hmm_layer, anc_probs_layer
    num_gpu = len([x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']) 
    if verbose:
        print("Using", num_gpu, "GPUs.")
    if num_gpu > 1:       
        
        #workaround: https://github.com/tensorflow/tensorflow/issues/50487
        import atexit
        mirrored_strategy = tf.distribute.MirroredStrategy()    
        atexit.register(mirrored_strategy._extended._collective_ops._pool.close) # type: ignore
        
        with mirrored_strategy.scope():
            model, msa_hmm_layer, anc_probs_layer = make_and_compile()
    else:
         model, msa_hmm_layer, anc_probs_layer = make_and_compile()
    steps = max(30, int(250*np.sqrt(indices.shape[0])/batch_size))
    callbacks = [tf.keras.callbacks.TerminateOnNaN()]
    history = model.fit(make_dataset(fasta_file, batch_size, True, indices), 
                          epochs=epochs,
                          steps_per_epoch=steps,
                          callbacks=callbacks,
                          verbose = 2*int(verbose))
    tf.get_logger().setLevel('INFO')
    return model, history