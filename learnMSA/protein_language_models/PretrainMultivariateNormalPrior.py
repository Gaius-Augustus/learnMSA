import argparse

parser = argparse.ArgumentParser(description="Trains a multivariate normal prior with full covariance matrix on PFAM embeddings.")
parser.add_argument("--lm", type=str, help="The language model to use. Supported values: proteinBERT, esm2, protT5.")
parser.add_argument("--batch", type=int, help="Batch size (number of sequence pairs) used during training.")
parser.add_argument("--epochs", type=int, help="The number of epochs to train. One epoch is always 1000 steps.")
parser.add_argument("--lr", type=float, default=1e-3, help="The learning rate.")
parser.add_argument("--gpu", type=str, default="0", help="The GPU to use.")
parser.add_argument("--max_len", type=int, default=400, help="Maximum sequence length. Longer proteins will be cropped.")
parser.add_argument("--name", type=str, default="", help="A name for the model. Per default it is named after the language model.")
parser.add_argument("--resume", action="store_true", help="Start training from an existing checkpoint.")

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
from learnMSA.protein_language_models.MultivariateNormalPrior import MultivariateNormalPrior, make_pdf_model



STEPS_PER_EPOCH = 1000

def make_embedding_model(encoder : Common.InputEncoder, language_model : Common.LanguageModel):
    """Utility function that constructs a keras model over a MultivariateNormalPrior. 
    """
    inputs = [tf.keras.Input(type_spec = spec) for spec in encoder.get_signature()]
    embeddings = language_model(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=[embeddings])
    return model

def make_full_model(emb_model, log_pdf_model):
    inputs = emb_model.inputs
    log_pdf = log_pdf_model(emb_model(inputs))
    model = tf.keras.Model(inputs=[inputs], outputs=[log_pdf])
    return model

def fit_model(encoder : Common.InputEncoder, emb_model, log_pdf_model, batch_size, max_len, lr, epochs, checkpoint_path):
    #get the full model
    model = make_full_model(emb_model, log_pdf_model)
    optimizer = tf.keras.optimizers.Adam(lr)
    model.compile(loss=lambda _, log_pdf: -log_pdf, optimizer=optimizer)
    clans_df, unique_clans, fasta_dict, clan_sizes, clan_families = data.get_clan_data(drop_overlap=False)
    num_clans = unique_clans.size
    ds = data.make_unsupervised_dataset(encoder, np.arange(num_clans), batch_size, max_len, fasta_dict, clan_sizes, clan_families)
    ds_y = tf.data.Dataset.from_tensor_slices(tf.zeros(1)).batch(batch_size).repeat()
    ds = tf.data.Dataset.zip((ds, ds_y))
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



if __name__ == "__main__":
    print(f"Maximum length is {args.max_len}. Will randomly crop sequences longer than this value.")
    print("Loading language model...")
    language_model, encoder = get_language_model(args.lm, args.max_len)
    emb_model = make_embedding_model(encoder, language_model)
    log_pdf_model = make_pdf_model(language_model.dim)
    name = args.name if args.name else args.lm
    checkpoint_path = f"priors/{name}/checkpoints"
    if args.resume:
        print("Resuming training...")
        log_pdf_model.load_weights(checkpoint_path)
    else:
        print("Starting training...")
    history = fit_model(encoder, emb_model, log_pdf_model, args.batch, args.max_len, 
                        lr=args.lr, epochs=args.epochs, checkpoint_path=checkpoint_path)
    history.history["lm"] = args.lm
    history.history["lr"] = args.lr
    history.history["batch"] = args.batch
    history.history["max_protein_len"] = args.max_len
    json.dump(history.history, open(f"priors/{name}/history.json", 'w'))