import argparse

parser = argparse.ArgumentParser(description="Trains a multivariate normal prior with full covariance matrix on PFAM embeddings.")
parser.add_argument("--lm", type=str, help="The language model to use. Supported values: proteinBERT, esm2, protT5.")
parser.add_argument("--batch", type=int, help="Batch size (number of sequence pairs) used during training.")
parser.add_argument("--gpu", type=str, default="0", help="The GPU to use.")
parser.add_argument("--max_len", type=int, default=400, help="Maximum sequence length. Longer proteins will be cropped.")
parser.add_argument("--name", type=str, default="", help="A name for the model. Per default it is named after the language model.")
parser.add_argument("--reduced", action="store_true", help="Train the prior on the reduced embedding space defined by a scoring model.")
parser.add_argument("--reduced_dim", type=int, default=32, help="Reduced embedding dimension used in the scoring model. Only relevant in reduced training mode.")
parser.add_argument("--components", type=int, default=1, help="Number of components in the multivariate normal mixture.")
parser.add_argument("--random_data", action="store_true", help="Shows the log pdf of the prior for random normal distributed embeddings with small variance and zero mean.")

args = parser.parse_args()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
import json
import sys
sys.path.append('../../')
import learnMSA.protein_language_models.Common as Common 
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


def get_language_model(name, max_len):
    if name == "proteinBERT":
        language_model, encoder = ProteinBERT.get_proteinBERT_model_and_encoder(max_len = max_len+2, trainable=False)
    elif name == "esm2":
        language_model, encoder = ESM2.ESM2LanguageModel(trainable=False), ESM2.ESM2InputEncoder()
    elif name == "esm2s":
        language_model, encoder = ESM2.ESM2LanguageModel(trainable=False, small=True), ESM2.ESM2InputEncoder(small=True)
    elif name == "protT5":
        language_model, encoder = ProtT5.ProtT5LanguageModel(trainable=False), ProtT5.ProtT5InputEncoder()
    else:
        raise ValueError(f"Language model {name} not supported.")
    return language_model, encoder


def get_scoring_layer(lm_name, dim):
    import learnMSA.protein_language_models as plm
    scoring_model = make_scoring_model(plm.common.dims[lm_name], 
                                        dim, 
                                        dropout=0.0, 
                                        trainable=False)
    scoring_model.load_weights(os.path.dirname(__file__)+f"/scoring_models_frozen/{lm_name}_{dim}/checkpoints")
    scoring_model.layers[-1].trainable = False #don't forget to freeze the scoring model!
    return scoring_model.layers[-1]


if __name__ == "__main__":
    # make the full model
    language_model, encoder = get_language_model(args.lm, args.max_len)
    emb_model = make_embedding_model(encoder, language_model)
    emb_dim = args.reduced_dim if args.reduced else language_model.dim
    log_pdf_model = make_pdf_model(emb_dim, components=args.components, aggregate_result=False)
    transform_emb_layer = get_scoring_layer(args.lm, args.reduced_dim) if args.reduced else None
    model = make_full_model(emb_model, log_pdf_model, transform_emb_layer)
    # load weights
    name = args.name if args.name else args.lm
    checkpoint_path = f"priors/{name}/checkpoints"
    log_pdf_model.load_weights(checkpoint_path)
    # compile and evaluate 
    if args.random_data:
        log_pdf_model.compile(loss=lambda match_mask, log_pdf: -aggregate(log_pdf * match_mask))
        steps = 100
        test_ds = data.make_random_data(emb_dim, args.batch, steps)
        r = log_pdf_model.evaluate(test_ds, steps=steps)
    else:
        model.compile(loss=lambda match_mask, log_pdf: -aggregate(log_pdf * match_mask))
        test_ds, steps = data.make_homfam_dataset(encoder, args.batch, homfam_path="/home/jovyan/brain/MSA-HMM-Analysis/data/homfam/refs/", for_prior=True)
        r = model.evaluate(test_ds, steps=steps-1)