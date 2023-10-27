import argparse

parser = argparse.ArgumentParser(description="Trains a multivariate normal prior with full covariance matrix on PFAM embeddings.")
parser.add_argument("--lm", type=str, help="The language model to use. Supported values: proteinBERT, esm2, protT5.")
parser.add_argument("--batch", type=int, default=64, help="Batch size (number of sequence pairs) used during training.")
parser.add_argument("--epochs", type=int, default=4, help="The number of epochs to train. One epoch is always 1000 steps.")
parser.add_argument("--lr", type=float, default=0.01, help="The learning rate.")
parser.add_argument("--gpu", type=str, default="0", help="The GPU to use.")
parser.add_argument("--max_len", type=int, default=512, help="Maximum sequence length. Longer proteins will be cropped.")
parser.add_argument("--name", type=str, default="", help="A name for the model. Per default it is named after the language model.")
parser.add_argument("--resume", action="store_true", help="Start training from an existing checkpoint.")
parser.add_argument("--activation", type=str, default="softmax", help="Activation function of the scoring model.")
parser.add_argument("--reduced_dim", type=int, default=32, help="Reduced embedding dimension used in the scoring model. Only relevant in reduced training mode.")
parser.add_argument("--suffix", type=str, default="", help="Suffix to identify the scoring model.")
parser.add_argument("--components", type=int, default=1, help="Number of components in the multivariate normal mixture.")

args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
import json
import sys
sys.path.append('../../')
import learnMSA.protein_language_models.Common as Common 
import learnMSA.protein_language_models.ProteinBERT as ProteinBERT
import learnMSA.protein_language_models.ESM2 as ESM2
import learnMSA.protein_language_models.ProtT5 as ProtT5
import learnMSA.protein_language_models.DataPipeline as data
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from learnMSA.protein_language_models.MultivariateNormalPrior import MultivariateNormalPrior, make_pdf_model, aggregate
from learnMSA.protein_language_models.BilinearSymmetric import make_scoring_model



STEPS_PER_EPOCH = 1000

def make_embedding_model(encoder : Common.InputEncoder, language_model : Common.LanguageModel):
    """Utility function that constructs a keras model over a MultivariateNormalPrior. 
    """
    inputs = [tf.keras.Input(type_spec = spec) for spec in encoder.get_signature()]
    embeddings = language_model(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=[embeddings])
    return model


def make_full_model(emb_model, log_pdf_model, transform_emb_layer=None):
    inputs = emb_model.inputs
    embeddings = emb_model(inputs)
    if not transform_emb_layer is None:
        embeddings = transform_emb_layer._reduce(embeddings, training=False)
    log_pdf = log_pdf_model(embeddings)
    model = tf.keras.Model(inputs=[inputs], outputs=[log_pdf])
    return model


def fit_model(encoder : Common.InputEncoder, emb_model, log_pdf_model, batch_size, max_len, lr, epochs, checkpoint_path, transform_emb_layer=None):
    #get the full model
    model = make_full_model(emb_model, log_pdf_model, transform_emb_layer)
    optimizer = tf.keras.optimizers.Adam(lr)
    model.compile(loss=lambda match_mask, log_pdf: -aggregate(log_pdf * match_mask), optimizer=optimizer)
    # drop the overlap to homfam, because we will use MSAs to decide which embeddings to take for training
    clans_df, unique_clans, fasta_dict, clan_sizes, clan_families = data.get_clan_data(drop_overlap=True)
    num_clans = unique_clans.size
    ds = data.make_column_prior_dataset(encoder, np.arange(num_clans), batch_size, max_len, fasta_dict, clan_sizes, clan_families)

    #ds_y = tf.data.Dataset.from_tensor_slices(tf.zeros(1)).batch(batch_size).repeat()
    #ds = tf.data.Dataset.zip((ds, ds_y))
    
    #custom callback tha only saves the prior, not the language model
    class PriorCheckpointCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            log_pdf_model.save_weights(checkpoint_path)
            print(f"Saved checkpoint {checkpoint_path}...")

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
    history = model.fit(ds, epochs=epochs, steps_per_epoch=STEPS_PER_EPOCH, callbacks=[PriorCheckpointCallback()])
    return history


def get_scoring_layer(config : Common.ScoringModelConfig):
    print(config)
    scoring_model = make_scoring_model(config, dropout=0.0, trainable=False)
    scoring_model_path = Common.get_scoring_model_path(config)
    scoring_model.load_weights(os.path.dirname(__file__)+"/"+scoring_model_path)
    scoring_model.layers[-1].trainable = False #don't forget to freeze the scoring model!
    return scoring_model.layers[-1]


if __name__ == "__main__":
    print(f"Maximum length is {args.max_len}. Will randomly crop sequences longer than this value.")
    print("Loading language model...")
    language_model, encoder = Common.get_language_model(args.lm, args.max_len, trainable=False)
    emb_model = make_embedding_model(encoder, language_model)
    config = Common.ScoringModelConfig(lm_name=args.lm, 
                                        dim=args.reduced_dim, 
                                        activation=args.activation,
                                        suffix=args.suffix)
    log_pdf_model = make_pdf_model(args.reduced_dim, 
                                    components=args.components,         
                                    aggregate_result=False)
    transform_emb_layer = get_scoring_layer(config) 
    name = args.name if args.name else args.lm
    checkpoint_path = os.path.dirname(__file__)+"/"+Common.get_prior_path(config, args.components)
    if args.resume:
        print("Resuming training...")
        log_pdf_model.load_weights(checkpoint_path)
    else:
        print("Starting training...")
    history = fit_model(encoder, emb_model, log_pdf_model, args.batch, args.max_len, 
                        lr=args.lr, epochs=args.epochs, checkpoint_path=checkpoint_path,
                       transform_emb_layer=transform_emb_layer)
    history.history["lm"] = args.lm
    history.history["lr"] = args.lr
    history.history["batch"] = args.batch
    history.history["max_protein_len"] = args.max_len
    history.history["reduced_dim"] = args.reduced_dim
    history.history["components"] = args.components
    history.history["activation"] = args.activation
    history.history["suffix"] = args.suffix
    json.dump(history.history, open(checkpoint_path[:-len("checkpoints/")]+"/history.json", 'w'))