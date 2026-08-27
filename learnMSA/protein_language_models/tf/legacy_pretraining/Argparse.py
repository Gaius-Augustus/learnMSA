import argparse

def make_scoring_model_argparser():
    parser = argparse.ArgumentParser(description="Trains a scoring model on the Pfam database with any families removed that have high similarity to a HomFam family.")
    parser.add_argument("--lm", type=str, help="The language model to use. Supported values: proteinBERT, esm2, protT5.")
    parser.add_argument("--lr", type=float, default=0.1, help="The learning rate.")
    parser.add_argument("--dim", type=int, help="The dimensionality of the reduced embedding space that is learned."
                                                "The scoring metric from which the probability of alignment is computed is based on vector pairs with this dimension.")
    parser.add_argument("--batch", type=int, default=64, help="Batch size (number of sequence pairs) used during training. If accumulation > 1, the batch size will be distributed among the accumulative updates.")
    parser.add_argument("--acc", type=int, default=1, help="Number of gradient accumulation steps.")
    parser.add_argument("--gpu", type=str, default="default", help="The GPU to use.")
    parser.add_argument("--finetune", action="store_true", help="Finetune and save to language model also. If not provided, the scoring model is fitted over the frozen language model.")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum sequence length. Longer proteins will be cropped.")
    parser.add_argument("--dropout", type=float, default=0.4, help="Maximum sequence length. Longer proteins will be cropped.")
    parser.add_argument("--activation", type=str, default="softmax", help="The activation function applied to dot-product scores of embedding pairs. Default: Softmax over the second sequence.")
    parser.add_argument("--suffix", type=str, default="", help="Suffix to identify the training run.")
    parser.add_argument("--weight0", type=float, default=1., help="Weight for the 0 class when using sigmoid activations and binary loss.")
    parser.add_argument("--weight1", type=float, default=1., help="Weight for the 1 class when using sigmoid activations and binary loss.")
    parser.add_argument("--resume", action="store_true", help="Start training from an existing checkpoint.")
    parser.add_argument("--unscaled", action="store_true", help="Used unscaled scores.")
    return parser

def make_mvn_prior_argparser():
    parser = argparse.ArgumentParser(description="Trains a multivariate normal prior with full covariance matrix on PFAM embeddings.")
    parser.add_argument("--lm", type=str, help="The language model to use. Supported values: proteinBERT, esm2, protT5.")
    parser.add_argument("--batch", type=int, default=64, help="Batch size (number of sequence pairs) used during training.")
    parser.add_argument("--epochs", type=int, default=4, help="The number of epochs to train. One epoch is always 1000 steps.")
    parser.add_argument("--lr", type=float, default=0.01, help="The learning rate.")
    parser.add_argument("--gpu", type=str, default="default", help="The GPU to use.")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum sequence length. Longer proteins will be cropped.")
    parser.add_argument("--resume", action="store_true", help="Start training from an existing checkpoint.")
    parser.add_argument("--activation", type=str, default="softmax", help="Activation function of the scoring model.")
    parser.add_argument("--reduced_dim", type=int, default=32, help="Reduced embedding dimension used in the scoring model. Only relevant in reduced training mode.")
    parser.add_argument("--suffix", type=str, default="", help="Suffix to identify the scoring model.")
    parser.add_argument("--components", type=int, default=1, help="Number of components in the multivariate normal mixture.")
    parser.add_argument("--unscaled", action="store_true", help="Used unscaled scores.")
    return parser