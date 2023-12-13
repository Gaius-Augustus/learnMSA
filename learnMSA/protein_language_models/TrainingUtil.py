import os
import sys
import tensorflow as tf
import numpy as np
import learnMSA.protein_language_models.DataPipeline as data
import learnMSA.protein_language_models.Common as common
from learnMSA.protein_language_models.BilinearSymmetric import make_scoring_model
from learnMSA.protein_language_models.MvnPrior import MvnPrior, aggregate




def make_masked_categorical(y_true, y_pred):
    """
    Preprocesses pairwise sequence data.
    Gets given labels of shape (b, L1, L2) and predictions of the same shape.
    Both dimensions L1 and L2 can contain padding and insertions.
    Returns labels and predictions of shape (k, L2) where k is the number of all first-sequence non-padding-or-insertion residues.
    Also returns a tensor of shape (k) with the sequence length of the first sequence of each pair for normalization.
    """
    mask = tf.reduce_any(tf.not_equal(y_true, 0), -1)
    # length of the first sequence, the padding of the second sequence does not matter
    norm = tf.reduce_sum(tf.cast(mask, y_true.dtype), -1, keepdims=True)
    norm = tf.math.maximum(norm, 1.)
    norm = tf.repeat(norm, tf.shape(mask)[-1], axis=-1)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    norm_masked = tf.boolean_mask(norm, mask)
    return y_true_masked, y_pred_masked, norm_masked
    

def make_masked_binary(y_true, y_pred):
    """
    Gets given labels of shape (b, L1, L2) and predictions of the same shape.
    Both dimensions L1 and L2 can contain padding or insertions.
    Returns labels and predictions of shape (k, 1) where k is the number of all residue pairs
    where neither of both is a padding or insertion residue.
    Also returns a tensor of shape (k) with product of both sequence lengths for each pair.
    """
    mask = tf.math.logical_and( tf.reduce_any(tf.not_equal(y_true, 0), -1, keepdims=True), 
                                tf.reduce_any(tf.not_equal(y_true, 0), -2, keepdims=True) )

    # the number of possible pairs
    norm1 = tf.reduce_sum(tf.cast(mask, y_true.dtype), -1, keepdims=True)
    norm2 = tf.reduce_sum(tf.cast(mask, y_true.dtype), -2, keepdims=True)
    norm = norm1 * norm2
    norm = tf.math.maximum(norm, 1.)
    y_true_masked = tf.boolean_mask(y_true, mask)[..., tf.newaxis]
    y_pred_masked = tf.boolean_mask(y_pred, mask)[..., tf.newaxis]
    norm_masked = tf.boolean_mask(norm, mask)
    return y_true_masked, y_pred_masked, norm_masked
    
    
def make_masked_func(func, categorical, name):
    """
    If categorical, the masked func is defined as func applied over the last dimension (seq 2) and averaged over all proper
    residues of seq 1. We define a residue as proper iff it is not a padding or insertion residue.
    If binary, the loss is defined as loss_func applied over the last dimension (seq 2) and averaged over all proper
    residue pairs. We define a residue pair as proper iff both residues are neither padding nor insertion residues.
    """
    def masked_loss(y_true, y_pred):
        if categorical:
            y_true_masked, y_pred_masked, norm_masked = make_masked_categorical(y_true, y_pred)
        else:
            y_true_masked, y_pred_masked, norm_masked = make_masked_binary(y_true, y_pred)
        cee = tf.cast(func(y_true_masked, y_pred_masked), y_true.dtype)
        cee /= norm_masked
        return tf.reduce_sum(cee) / tf.cast(tf.shape(y_true)[0], y_true.dtype)
    masked_loss.__name__ = name
    return masked_loss


def make_binary_loss(weight_0, weight_1):
    def weighted_binary_cross_entropy(y_true, y_pred):
        bee = tf.keras.metrics.binary_crossentropy(tf.expand_dims(y_true, -1), 
                                                   tf.expand_dims(y_pred, -1))
        bee *= tf.cast(y_true == 0, bee.dtype) * weight_0 + tf.cast(y_true == 1, bee.dtype) * weight_1
        return tf.reduce_mean(bee, -1)
    return weighted_binary_cross_entropy



def make_full_scoring_model(encoder : common.InputEncoder, 
               language_model : common.LanguageModel, 
               scoring_model_config : common.ScoringModelConfig, 
               dropout, 
               trainable=True):
    """Constructs a full scoring model with a given encoder and language model that maps tokens to alignment probabilities.
    """
    #the input specs depend on the encoder
    inputs1 = [tf.keras.Input(type_spec = spec) for spec in encoder.get_signature()]
    inputs2 = [tf.keras.Input(type_spec = spec) for spec in encoder.get_signature()]
    #embed query and target
    emb1 = language_model(inputs1)
    emb2 = language_model(inputs2)
    #score the embeddings
    scoring_model = make_scoring_model(scoring_model_config, dropout, trainable=trainable)
    output = scoring_model([emb1, emb2])
    # construct a model and compile for a standard binary classification task
    model = tf.keras.models.Model(inputs=[inputs1, inputs2], outputs=output)
    return model, scoring_model


def get_loss_and_metrics(scoring_model_config : common.ScoringModelConfig,
                        weight0, weight1):
    # set up the correct loss and metrics depending on the setting (categorical, binary)
    if scoring_model_config.activation == "softmax":
        loss_func = tf.keras.metrics.categorical_crossentropy
        acc_func = tf.keras.metrics.categorical_accuracy
        other_metrics = ["categorical_accuracy"]
    elif scoring_model_config.activation == "sigmoid":
        loss_func = make_binary_loss(weight_0 = weight0 , weight_1 = weight1)
        acc_func = tf.keras.metrics.binary_accuracy
        other_metrics = [tf.keras.metrics.Precision(), 
                         tf.keras.metrics.Recall()]
    else:
        print(f"Unknown activation {scoring_model_config.activation}.")
        sys.exit()
    return loss_func, acc_func, other_metrics
    

def compile_scoring_model(model, loss_func, acc_func, categorical, accumulate_gradients=1, lr=1e-3, other_metrics=[]):
    optimizer = tf.keras.optimizers.Adam(lr)
    convert_to_accumulate_gradient_optimizer(optimizer, accumulate_gradients)
    model.compile(loss=make_masked_func(loss_func, categorical, "loss"), 
                  optimizer=optimizer,
                  metrics=[make_masked_func(acc_func, categorical, "acc")] + other_metrics)



def fit_scoring_model(encoder : common.InputEncoder, 
                        model, 
                        loss_func, 
                        acc_func,
                        categorical, 
                        batch_size, 
                        accumulate_gradients,
                        max_len, 
                        lr, 
                        callbacks=[], 
                        other_metrics=[],
                        steps_per_epoch=1000, 
                        epochs=4):
    compile_scoring_model(model, loss_func, acc_func, categorical, accumulate_gradients, lr, other_metrics)
    print(f"Accumulates gradients over {accumulate_gradients} steps with a batch size of {batch_size//accumulate_gradients} each.") 
    clans_df, unique_clans, fasta_dict, clan_sizes, clan_families = data.get_clan_data(drop_overlap=True)
    num_clans = unique_clans.size
    ds = data.make_dataset(encoder, np.arange(num_clans), batch_size//accumulate_gradients, max_len, fasta_dict, clan_sizes, clan_families)
    history = model.fit(ds, epochs=epochs, steps_per_epoch=int(steps_per_epoch * accumulate_gradients), callbacks=callbacks) 
    return history






def make_embedding_model(encoder : common.InputEncoder, language_model : common.LanguageModel):
    """Constructs a model that maps a sequence to its embedding.
    """
    inputs = [tf.keras.Input(type_spec = spec) for spec in encoder.get_signature()]
    embeddings = language_model(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=[embeddings])
    return model



def get_scoring_layer(config : common.ScoringModelConfig):
    scoring_model = make_scoring_model(config, dropout=0.0, trainable=False)
    scoring_model_path = common.get_scoring_model_path(config)
    scoring_model.load_weights(os.path.dirname(__file__)+"/"+scoring_model_path)
    scoring_model.layers[-1].trainable = False
    return scoring_model.layers[-1]


def make_full_mvn_prior_model(emb_model, log_pdf_model, scoring_layer):
    """Constructs a model that maps a sequence to its embedding and then to a log probability density.
    """
    inputs = emb_model.inputs
    embeddings = emb_model(inputs)
    embeddings = scoring_layer._reduce(embeddings, training=False)
    log_pdf = log_pdf_model(embeddings)
    model = tf.keras.Model(inputs=[inputs], outputs=[log_pdf])
    return model


def make_reduced_emb_model(emb_model, scoring_layer):
    """Constructs a helper model that computes embeddings in the reduced space.
    """
    inputs = emb_model.inputs
    embeddings = emb_model(inputs)
    embeddings = scoring_layer._reduce(embeddings, training=False)
    model = tf.keras.Model(inputs=[inputs], outputs=[embeddings])
    return model


def fit_mvn_prior(encoder : common.InputEncoder, 
                emb_model, 
                log_pdf_model, 
                scoring_layer,
                batch_size, 
                max_len, 
                lr, 
                epochs, 
                checkpoint_path, 
                steps_per_epoch=1000):
    #get the full model
    model = make_full_mvn_prior_model(emb_model, log_pdf_model, scoring_layer)
    optimizer = tf.keras.optimizers.Adam(lr)
    model.compile(loss=lambda match_mask, log_pdf: -aggregate(log_pdf, match_mask), optimizer=optimizer)
    # drop the overlap to homfam, because we will use MSAs to decide which embeddings to take for training
    clans_df, unique_clans, fasta_dict, clan_sizes, clan_families = data.get_clan_data(drop_overlap=True)
    num_clans = unique_clans.size
    ds = data.make_column_prior_dataset(encoder, np.arange(num_clans), batch_size, max_len, fasta_dict, clan_sizes, clan_families)
    #custom callback tha only saves the prior, not the language model
    class PriorCheckpointCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            log_pdf_model.save_weights(checkpoint_path)
            print(f"Saved checkpoint {checkpoint_path}...")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
    history = model.fit(ds, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=[PriorCheckpointCallback()])
    return history


class LinearWarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, step_scale=1., warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.step_scale = step_scale
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32) * self.step_scale
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)



# accumulate gradients to account for low batch sizes
# see https://github.com/keras-team/keras/issues/3556
def convert_to_accumulate_gradient_optimizer(orig_optimizer, update_params_frequency):
    if update_params_frequency < 1:
        raise ValueError('update_params_frequency must be >= 1')
    orig_get_gradients = orig_optimizer.get_gradients
    orig_get_updates = orig_optimizer.get_updates
    accumulated_iterations = tf.keras.backend.variable(0, dtype='int64', name='accumulated_iterations')
    orig_optimizer.accumulated_iterations = accumulated_iterations

    def updated_get_gradients(self, loss, params):
        return self.accumulate_gradient_accumulators

    def updated_get_updates(self, loss, params):
        self.accumulate_gradient_accumulators = [tf.keras.backend.zeros(tf.keras.backend.int_shape(p), dtype=tf.keras.backend.dtype(p)) for p in params]
        updates_accumulated_iterations = tf.keras.backend.update_add(accumulated_iterations, 1)
        new_grads = orig_get_gradients(loss, params)
        #average over accumulations
        new_grads = [g / tf.keras.backend.cast(update_params_frequency, tf.keras.backend.dtype(g)) for g in new_grads]
        self.updated_grads = [tf.keras.backend.update_add(p, g) for p, g in zip(self.accumulate_gradient_accumulators, new_grads)]
        def update_function():
            with tensorflow.control_dependencies(orig_get_updates(loss, params)):
                reset_grads = [tf.keras.backend.update(p, tf.keras.backend.zeros(tf.keras.backend.int_shape(p), dtype=tf.keras.backend.dtype(p))) for p in self.accumulate_gradient_accumulators]
            return tensorflow.group(*(reset_grads + [updates_accumulated_iterations]))
        def just_store_function():
            return tensorflow.group(*[updates_accumulated_iterations])
        
        update_switch = tf.keras.backend.equal((updates_accumulated_iterations) % update_params_frequency, 0)
        
        with tensorflow.control_dependencies(self.updated_grads):
            self.updates = [tf.keras.backend.switch(update_switch, update_function, just_store_function)]
            return self.updates

    orig_optimizer.get_gradients = updated_get_gradients.__get__(orig_optimizer, type(orig_optimizer))
    orig_optimizer.get_updates = updated_get_updates.__get__(orig_optimizer, type(orig_optimizer))