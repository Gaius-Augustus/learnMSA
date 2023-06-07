*Our tool is under active development and feedback is very much appreciated.*

# learnMSA: Learning and Aligning large Protein Families

# Introduction
Multiple sequence alignment formulated as a statistical machine learning problem, where an optimal profile hidden Markov model for a potentially ultra-large family of protein sequences is searched and an alignment is decoded. We use an automatically differentiable variant of the Forward algorithm to align via gradient descent.

## Features

- Aligns large numbers of protein sequences with state-of-the-art accuracy
- Enables ultra-large alignment of millions of sequences 
- Scales linear in the number of sequences (does not require a guide tree)
- Memory efficient (depending on sequence length, aligning millions of sequences on a laptop is possible)
- (optional) GPU acceleration 
- Visualize a profile HMM or a sequence logo of the consensus motif

## Current limitations

- Requires many sequences (in most cases starting at 1000, a few 100 might still be enough) to achieve state-of-the-art accuracy
- Only for protein sequences
- Increasingly slow for long proteins with a length > 1000 residues

# Installation

Choose according to your preference:

## Using Bioconda
  
  If you haven't done it yet, set up [Bioconda channels](https://bioconda.github.io/) first.
  
  We recommend to install in a clean environment:
  
  <code>conda create -n learnMSA learnMSA</code>
  
  which creates an environment called learnMSA and installs the package learnMSA.
  
  To run learnMSA, you have to activate the environment first:
  
  <code>conda activate learnMSA</code>.
  
  We recommend to use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to have a clean base environment.

## Using pip

  <code>pip install learnMSA</code>
  
## With GPU support

*Optional, but recommended for proteins longer than 100 residues.*

You have to meet the [TensorFlow GPU](https://www.tensorflow.org/install/gpu) requirements. The install instructions above are sufficient if the cudnn and cudatoolkit packages are installed on your system.

LearnMSA will notify you whether it finds any GPUs it can use or it will fall back to CPU.

## Command line use after installing with Bioconda or pip

<code>learnMSA -i INPUT_FILE -o OUTPUT_FILE</code>
  
<code>learnMSA -h</code>

## Manual installation

Requirements:
- [TensorFlow](https://github.com/tensorflow/tensorflow) (tested versions: 2.5, >=2.7)
- [networkx](https://networkx.org/) 
- [logomaker](https://logomaker.readthedocs.io/en/latest/) 
- python 3.9 (there are known issues with 3.7 which is deprecated and 3.8 is untested)

1. Clone the repository: 

  <code>git clone https://github.com/Gaius-Augustus/learnMSA</code>
  
2. Install dependencies:

  <code>pip install tensorflow logomaker networkx</code>
  
3. Run:

  <code>cd learnMSA</code>
  
  <code>python3 learnMSA.py --help</code>
  

## Interactive notebook with visualization:

Run the notebook <code>MsaHmm.ipynb</code> with juypter.

# Version 1.1.0 improvements

- Parallel training of multiple models and reduced memory footprint (train more models in less time)
- Customize learnMSA via code (e.g. by changing emission type, prior or the number of rate matricies used to compute ancestral probabilities)

![alt text](https://github.com/Ung0d/MSA-HMM-Analysis/blob/main/fig/boxplots_sp_homfam.png?raw=true)
![alt text](https://github.com/Ung0d/MSA-HMM-Analysis/blob/main/fig/learnMSA_fast_comparison.png?raw=true)
![alt text](https://github.com/Ung0d/MSA-HMM-Analysis/blob/main/fig/learnMSA_fast_comparison_large.png?raw=true)

# Publications

Becker F, Stanke M. **learnMSA: learning and aligning large protein families**. *GigaScience*. 2022
