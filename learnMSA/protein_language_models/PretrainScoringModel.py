import argparse

parser = argparse.ArgumentParser(description="Trains a scoring model on the Pfam database with any families removed that have high similarity to a HomFam family.")
parser.add_argument("--lm", type=str, help="The language model to use. Supported values: proteinBERT, esm2, protT5.")
parser.add_argument("--dim", type=int, help="The dimensionality of the reduced embedding space that is learned."
                                            "The scoring metric from which the probability of alignment is computed is based on vector pairs with this dimension.")
parser.add_argument("--batch", type=int, help="Batch size (number of sequence pairs) used during training. If accumulation > 1, the batch size will be distributed among the accumulative updates.")
parser.add_argument("--acc", type=int, default=1, help="Number of gradient accumulation steps.")
parser.add_argument("--gpu", type=str, default="0", help="The GPU to use.")
parser.add_argument("--finetune", action="store_true", help="Finetune and save to language model also. If not provided, the scoring model is fitted over the frozen language model.")
parser.add_argument("--max_len", type=int, default=400, help="Maximum sequence length. Longer proteins will be cropped.")
parser.add_argument("--dropout", type=float, default=0.4, help="Maximum sequence length. Longer proteins will be cropped.")
parser.add_argument("--activation", type=str, default="softmax", help="The activation function applied to dot-product scores of embedding pairs. Default: Softmax over the second sequence.")
parser.add_argument("--suffix", type=str, default="", help="Suffix to identify the training run.")
parser.add_argument("--weight0", type=float, default=1., help="Weight for the 0 class when using sigmoid activations and binary loss.")
parser.add_argument("--weight1", type=float, default=1., help="Weight for the 1 class when using sigmoid activations and binary loss.")

args = parser.parse_args()


import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
import json
import sys
sys.path.append('../../')
import Common as common
import DataPipeline as data
from BilinearSymmetric import make_scoring_model
import ProteinBERT
import ESM2
import ProtT5 
import tensorflow as tf
import numpy as np



def make_masked(y_true, y_pred):
    mask = tf.reduce_any(tf.not_equal(y_true, 0), -1)
    seq_lens = tf.reduce_sum(tf.cast(mask, y_true.dtype), -1, keepdims=True)
    seq_lens = tf.math.maximum(seq_lens, 1.)
    seq_lens = tf.repeat(seq_lens, tf.shape(mask)[-1], axis=-1)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    seq_lens_masked = tf.boolean_mask(seq_lens, mask)
    return y_true_masked, y_pred_masked, seq_lens_masked
    
    
def make_masked_loss(loss_func):
    def masked_loss(y_true, y_pred):
        y_true_masked, y_pred_masked, seq_lens_masked = make_masked(y_true, y_pred)
        cee = loss_func(y_true_masked, y_pred_masked)
        cee /= seq_lens_masked
        return tf.reduce_sum(cee) / tf.cast(tf.shape(y_true)[0], y_true.dtype)
    return masked_loss


# Average accuracy per sequence of correctly aligned query-target residues 
# not counting insertions or padding.
def make_masked_acc(acc_func):
    def masked_accuracy(y_true, y_pred):
        y_true_masked, y_pred_masked, seq_lens_masked = make_masked(y_true, y_pred)
        acc = acc_func(y_true_masked, y_pred_masked)
        acc /= seq_lens_masked
        return tf.reduce_sum(acc) / tf.cast(tf.shape(y_true)[0], y_true.dtype)
    return masked_accuracy


def make_binary_loss(weight_0, weight_1):
    def weighted_binary_cross_entropy(y_true, y_pred):
        bee = tf.keras.metrics.binary_crossentropy(tf.expand_dims(y_true, -1), 
                                                   tf.expand_dims(y_pred, -1))
        bee *= tf.cast(y_true == 0, bee.dtype) * weight_0 + tf.cast(y_true == 1, bee.dtype) * weight_1
        return tf.reduce_mean(bee, -1)
    return weighted_binary_cross_entropy


def make_full_model(encoder : common.InputEncoder, 
               language_model : common.LanguageModel, 
               reduced_dim, dropout, activation):
    #the input specs depend on the encoder
    inputs1 = [tf.keras.Input(type_spec = spec) for spec in encoder.get_signature()]
    inputs2 = [tf.keras.Input(type_spec = spec) for spec in encoder.get_signature()]
    #embed query and target
    emb1 = language_model(inputs1)
    emb2 = language_model(inputs2)
    #score the embeddings
    scoring_model = make_scoring_model(language_model.dim, reduced_dim, dropout, trainable=True, activation=activation)
    output = scoring_model([emb1, emb2])
    # construct a model and compile for a standard binary classification task
    model = tf.keras.models.Model(inputs=[inputs1, inputs2], outputs=output)
    return model, scoring_model


def get_language_model(name, max_len, trainable):
    if name == "proteinBERT":
        language_model, encoder = ProteinBERT.get_proteinBERT_model_and_encoder(max_len = max_len+2, trainable=trainable)
    elif name == "esm2":
        language_model, encoder = ESM2.ESM2LanguageModel(trainable=trainable), ESM2.ESM2InputEncoder()
    elif name == "esm2s":
        language_model, encoder = ESM2.ESM2LanguageModel(trainable=trainable, small=True), ESM2.ESM2InputEncoder(small=True)
    elif name == "protT5":
        language_model, encoder = ProtT5.ProtT5LanguageModel(trainable=trainable), ProtT5.ProtT5InputEncoder()
    else:
        raise ValueError(f"Language model {name} not supported.")
    return language_model, encoder


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
    

def compile_model(model, loss_func, acc_func, accumulate_gradients=1, lr=1e-3, other_metrics=[]):
    optimizer = tf.keras.optimizers.Adam(lr)
    convert_to_accumulate_gradient_optimizer(optimizer, accumulate_gradients)
    model.compile(loss=make_masked_loss(loss_func), 
                  optimizer=optimizer,
                  metrics=[make_masked_acc(acc_func)] + other_metrics)


def fit_model(encoder : common.InputEncoder, model, loss_func, acc_func, batch_size, accumulate_gradients, max_len, lr, callbacks=[], other_metrics=[]):
    compile_model(model, loss_func, acc_func, accumulate_gradients, lr, other_metrics)
    print(f"Accumulates gradients over {accumulate_gradients} steps with a batch size of {batch_size//accumulate_gradients} each.") 
    clans_df, unique_clans, fasta_dict, clan_sizes, clan_families = data.get_clan_data(drop_overlap=True)
    num_clans = unique_clans.size
    ds = data.make_dataset(encoder, np.arange(num_clans), batch_size//accumulate_gradients, max_len, fasta_dict, clan_sizes, clan_families)
    history = model.fit(ds, epochs=35, steps_per_epoch=int(1000 * accumulate_gradients), callbacks=callbacks) 
    return history


if __name__ == "__main__":
    print("Finetuning:", args.finetune)
    print(f"Maximum length is {args.max_len}. Will randomly crop sequences longer than this value.")
    print(tf.config.list_physical_devices('GPU'))
    language_model, encoder = get_language_model(args.lm, args.max_len, trainable=args.finetune)
    if args.activation == "softmax":
        activation = tf.nn.softmax
        loss_func = tf.keras.metrics.categorical_crossentropy
        acc_func = tf.keras.metrics.categorical_accuracy
        other_metrics = ["categorical_accuracy"]
    elif args.activation == "sigmoid":
        activation = tf.math.sigmoid
        loss_func = make_binary_loss(weight_0 = args.weight0 , weight_1 = args.weight1)
        acc_func = tf.keras.metrics.binary_accuracy
        other_metrics = [tf.keras.metrics.Precision(), 
                         tf.keras.metrics.Recall()]
    else:
        print(f"Unknown activation {args.activation}.")
        sys.exit()
    full_model, scoring_model = make_full_model(encoder, language_model, reduced_dim = args.dim, dropout = args.dropout, activation = activation)
    #compile with a learning rate schedule that decays over time
    lr_schedule = LinearWarmupSchedule(language_model.dim, step_scale = 1/args.acc)
    
    #custom callback tha only saves the prior, not the language model
    class PriorCheckpointCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if args.finetune:
                scoring_model.save_weights(f"scoring_models/{args.lm}_{args.dim}_{args.activation}{args.suffix}/checkpoints")
                language_model.model.save_weights(f"finetuned_models/{args.lm}_{args.dim}_{args.activation}{args.suffix}/checkpoints")
            else:
                scoring_model.save_weights(f"new_scoring_models_frozen/{args.lm}_{args.dim}_{args.activation}{args.suffix}/checkpoints")
            print(f"Saved checkpoint...")
            
    history = fit_model(encoder, full_model, loss_func, acc_func, batch_size=args.batch, 
                        accumulate_gradients=args.acc, max_len=args.max_len, lr=lr_schedule,
                        callbacks=[PriorCheckpointCallback()], other_metrics=other_metrics)
    if args.finetune:
        json_path = f"scoring_models/{args.lm}_{args.dim}_{args.activation}{args.suffix}/history.json"
    else:
        json_path = f"new_scoring_models_frozen/{args.lm}_{args.dim}_{args.activation}{args.suffix}/history.json"
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    json.dump(history.history, open(json_path, 'w'))