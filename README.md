*Our tool is under active development and feedback is very much appreciated.*

# learnMSA: Learning and Aligning large Protein Families

<img src="https://github.com/Gaius-Augustus/learnMSA/blob/main/logo/training_loop.gif" alt="" loop=infinite>

# Introduction
Multiple sequence alignment formulated as a statistical machine learning problem, where an optimal profile hidden Markov model for a potentially ultra-large family of protein sequences is learned from unaligned sequences and an alignment is decoded. We use a novel, automatically differentiable variant of the forward algorithm to train pHMMs via gradient descent.

Since version 2.0.0, learnMSA can utilize protein language models (`--use_language_model`) for significantly improved accuracy.

## Features

- Aligns large numbers of protein sequences with state-of-the-art accuracy
- Can utilize protein language models (`--use_language_model`) for significantly improved accuracy
- Enables ultra-large alignment of millions of sequences 
- GPU acceleration, multi-GPU support
- Scales linear in the number of sequences (does not require a guide tree)
- Memory efficient (depending on sequence length, aligning millions of sequences on a laptop is possible)
- Visualize a profile HMM or a sequence logo of the consensus motif

## Current limitations

- Requires many sequences (in most cases starting at 1000, a few 100 might still be enough) to achieve state-of-the-art accuracy
- Only for protein sequences
- Increasingly slow for long proteins with a length > 1000 residues

# Installation

Currently, learnMSA 2.0.0 is only supported for python <= 3.10, because of incompatibilities with TensorFlow. This is expected to change in the future.

*Recommended way to install learnMSA:*

```
conda create -n learnMSA_env python=3.10 pip
conda activate learnMSA_env
pip install learnMSA
```

Another quick option is to use mamba (if conda is too slow):

If you haven't done it yet, set up [Bioconda channels](https://bioconda.github.io/) first.

```
conda install mamba
mamba create -n learnMSA_env learnMSA
conda activate learnMSA_env
```

or simply with conda:

```
conda create -n learnMSA_env learnMSA
conda activate learnMSA_env
```

While in principle we attempt to support all tensorflow versions since 2.5.0, there are known incompatiblities with tf >= 2.12.0. We recommend tensorflow version 2.10.0 if there are no particular reasons to use something else.

## Command line use after installing with Bioconda or pip

<code>learnMSA -h</code>

Align with language model support (recommended):

<code>learnMSA -i INPUT_FILE -o OUTPUT_FILE --use_language_model</code>

Without language model support:

<code>learnMSA -i INPUT_FILE -o OUTPUT_FILE</code>

We always recommend to run learnMSA with the `--sequence_weights` flag to improve accuracy. This requires `mmseqs2` to be installed. You can use conda for this: `conda install -c bioconda mmseqs2`. Sequence weights (and thus the `mmseqs2` requirement are turned off by default).

*Since learnMSA version 1.2.0, insertions are aligned with famsa. This improves overall accuracy. The old behavior can be restored with the `--unaligned_insertions` flag.*

To output a pdf with a sequence logo alongside the msa, use `--logo`. For a fun gif that visualizes the training process, you can use `--logo_gif` (attention, slows down training and should not be used for real alignments).
  
## Interactive notebook with visualization:

Run the notebooks <code>learnMSA_demo.ipynb</code> or <code>learnMSA_with_language_model_demo.ipynb</code> with juypter.
  
## Benchmark:

![alt text](https://github.com/felbecker/snakeMSA/blob/main/plots/barplots.png?raw=true)

# Publications

Becker F, Stanke M. **learnMSA: learning and aligning large protein families**. *GigaScience*. 2022
