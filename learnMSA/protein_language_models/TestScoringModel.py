import argparse

parser = argparse.ArgumentParser(description="Trains a scoring model on the Pfam database with any families removed that have high similarity to a HomFam family.")
parser.add_argument("--lm", type=str, help="The language model to use. Supported values: proteinBERT, esm2, protT5.")
parser.add_argument("--dim", type=int, help="The dimensionality of the reduced embedding space that is learned."
                                            "The scoring metric from which the probability of alignment is computed is based on vector pairs with this dimension.")
parser.add_argument("--batch", type=int, default=256, help="Batch size (number of sequence pairs) used during training.")
parser.add_argument("--gpu", type=str, default="0", help="The GPU to use.")
parser.add_argument("--max_len", type=int, default=1000, help="Maximum sequence length. Longer proteins will be cropped.")
parser.add_argument("--dropout", type=float, default=0.4, help="Maximum sequence length. Longer proteins will be cropped.")
parser.add_argument("--activation", type=str, default="softmax", help="The activation function applied to dot-product scores of embedding pairs. Default: Softmax over the second sequence.")
parser.add_argument("--suffix", type=str, default="", help="Suffix to identify the training run.")
parser.add_argument("--weight0", type=float, default=1., help="Weight for the 0 class when using sigmoid activations and binary loss.")
parser.add_argument("--weight1", type=float, default=1., help="Weight for the 1 class when using sigmoid activations and binary loss.")

args = parser.parse_args()


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
               config : common.ScoringModelConfig,
               dropout):
    #the input specs depend on the encoder
    inputs1 = [tf.keras.Input(type_spec = spec) for spec in encoder.get_signature()]
    inputs2 = [tf.keras.Input(type_spec = spec) for spec in encoder.get_signature()]
    #embed query and target
    emb1 = language_model(inputs1)
    emb2 = language_model(inputs2)
    #score the embeddings
    scoring_model = make_scoring_model(config, dropout, trainable=False)
    output = scoring_model([emb1, emb2])
    # construct a model and compile for a standard binary classification task
    model = tf.keras.models.Model(inputs=[inputs1, inputs2], outputs=output)
    return model, scoring_model


if __name__ == "__main__":
    language_model, encoder = common.get_language_model(args.lm, args.max_len, trainable=False)
    if args.activation == "softmax":
        loss_func = tf.keras.metrics.categorical_crossentropy
        acc_func = tf.keras.metrics.categorical_accuracy
        other_metrics = ["categorical_accuracy"]
    elif args.activation == "sigmoid":
        loss_func = make_binary_loss(weight_0 = args.weight0 , weight_1 = args.weight1)
        acc_func = tf.keras.metrics.binary_accuracy
        other_metrics = [tf.keras.metrics.Precision(), 
                         tf.keras.metrics.Recall()]
    else:
        print(f"Unknown activation {args.activation}.")
        sys.exit()
    config = common.ScoringModelConfig(lm_name=args.lm, 
                                        dim=args.dim, 
                                        activation=args.activation, 
                                        suffix=args.suffix)
    full_model, scoring_model = make_full_model(encoder, language_model, config, args.dropout)
    model_path = common.get_scoring_model_path(config)
    scoring_model.load_weights(model_path)
    full_model.compile(loss=make_masked_loss(loss_func), metrics=[make_masked_acc(acc_func)] + other_metrics)
    test_ds, steps = data.make_homfam_dataset(encoder, args.batch, homfam_path="../../../../MSA-HMM-Analysis/data/homfam/refs/")
    r = full_model.evaluate(test_ds, steps=steps-1, verbose=0)
    print(r)