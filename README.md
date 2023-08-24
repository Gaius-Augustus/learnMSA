*Our tool is under active development and feedback is very much appreciated.*

# learnMSA: Learning and Aligning large Protein Families

# Introduction
Multiple sequence alignment formulated as a statistical machine learning problem, where an optimal profile hidden Markov model for a potentially ultra-large family of protein sequences is learned from unaligned sequences and an alignment is decoded. We use a novel, automatically differentiable variant of the Forward algorithm to train pHMMs via gradient descent.

## Features

- Aligns large numbers of protein sequences with state-of-the-art accuracy
- Enables ultra-large alignment of millions of sequences 
- GPU acceleration, multi-GPU support
- Scales linear in the number of sequences (does not require a guide tree)
- Memory efficient (depending on sequence length, aligning millions of sequences on a laptop is possible)
- Visualize a profile HMM or a sequence logo of the consensus motif
- Experimental use of large protein language models to improve alignment accuracy

## Current limitations

- Requires many sequences (in most cases starting at 1000, a few 100 might still be enough) to achieve state-of-the-art accuracy
- Only for protein sequences
- Increasingly slow for long proteins with a length > 1000 residues

# Installation

Choose according to your preference:

## Using Bioconda
  
  If you haven't done it yet, set up [Bioconda channels](https://bioconda.github.io/) first.

  *Recommended way to install learnMSA:*

  ```
  conda install mamba
  mamba create -n learnMSA_env learnMSA
  ```
  
  which creates an environment called `learnMSA_env` and installs learnMSA in it.
  
  To run learnMSA, you have to activate the environment first:
  
  <code>conda activate learnMSA_env</code>.

  While in principle we attempt to support all tensorflow versions since 2.5.0, there are known incompatiblities with tf >= 2.12.0. We recommend tensorflow version 2.10.0 if there are no particular reasons to use something else.

## Using pip
  
  <code>pip install learnMSA</code>
  
## With GPU support

*Optional, but recommended for proteins longer than 100 residues. The install instructions above may already be sufficient to support GPU depending on your system. LearnMSA will notify you whether it finds any GPUs it can use or it will fall back to CPU.*

You have to meet the [TensorFlow GPU](https://www.tensorflow.org/install/gpu) requirements and may do the cuda setup steps.

## Command line use after installing with Bioconda or pip

<code>learnMSA -i INPUT_FILE -o OUTPUT_FILE</code>
  
<code>learnMSA -h</code>

*Since learnMSA version 1.2.0, insertions are aligned with famsa. This improves overall accuracy. The old behavior can be restored with the `--unaligned_insertions` flag.*

## Manual installation

Requirements:
- [TensorFlow](https://github.com/tensorflow/tensorflow) (we recommend 2.10.0, tested versions: 2.5, >=2.7)
- [networkx](https://networkx.org/) 
- [logomaker](https://logomaker.readthedocs.io/en/latest/)
- [seaborn](https://seaborn.pydata.org/)
- [biopython](https://biopython.org/) (>=1.69)
- [pyfamsa](https://pypi.org/project/pyfamsa/)
- [transformers](https://huggingface.co/docs/transformers/index)
- python 3.9 (there are known issues with 3.7 which is deprecated and 3.8 is untested)

1. Clone the repository 

  <code>git clone https://github.com/Gaius-Augustus/learnMSA</code>
  
2. Install dependencies with pip or conda
  
3. Run

  <code>cd learnMSA</code>
  
  <code>python3 learnMSA.py --help</code>
  

## Interactive notebook with visualization:

Run the notebooks <code>learnMSA_demo.ipynb</code> or <code>learnMSA_with_language_model_demo.ipynb</code> with juypter.

# Version 1.3.0 improvements

- Use `pyfamsa` to align insertions, also made aligning insertions the default behavior (also added `--unaligned_insertions` flag).
- Use `biopython` for data parsing. Many more input file formats are now available as well as the experimental `indexed_data` flag for large datasets that allows constant memory model training. 
- Multi GPU training works now. It is mostly beneficial for large datasets with long sequences. It can negatively affect performance otherwise.
- Added the experimental `--use_language_model` flag that uses a large, pretrained protein language model to guide the MSA and improve alignment accuracy.
  
![alt text](https://github.com/felbecker/snakeMSA/blob/main/plots/SP_TC.png?raw=true)

# Version 1.2.0 improvements

- insertions that were left unaligned by learnMSA can now be aligned retroactively by a third party aligner which improves accuracy on the HomFam benchmark by about 2%-points

# Version 1.1.0 improvements

- Parallel training of multiple models and reduced memory footprint (train more models in less time)
- Customize learnMSA via code (e.g. by changing emission type, prior or the number of rate matricies used to compute ancestral probabilities)

# Publications

Becker F, Stanke M. **learnMSA: learning and aligning large protein families**. *GigaScience*. 2022
