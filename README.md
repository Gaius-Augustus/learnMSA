# learnMSA: Learning and Aligning large Protein Families

## Introduction
Multiple sequence alignment formulated as a statistical machine learning problem, where an optimal profile hidden Markov model for a potentially very large family of protein sequences is searched and an alignment is decoded. We take advantage of TensorFlows automatic differentiation to differentiate the forward algorithm, i.e. compute gradients over all possible alignments.

![alt text](https://github.com/Ung0d/MSA-HMM-Analysis/blob/main/fig/boxplots_sp_homfam.png?raw=true)

On ultra-large (millions of sequences) data we produce alignments with better accuracy and are (occasionally multiple times) faster than state of the art aligners. We think that our tool can be a first step towards a future-proof framework for ultra large MSA that lends itself to a variety of opportunities for further improvements.

***When should I use it?***

Our primary focus is on large numbers of protein sequences (starting at 10.000) with reasonable length. The tool is under active development. 


## Installation

#### Using pip:

  <code>pip install learnMSA</code>
  
This installs learnMSA as a python package with all dependencies and also enables command line use:

  <code>learnMSA -i INPUT_FILE -o OUTPUT_FILE</code>
  
  <code>learnMSA -h</code>


#### Manual installation:

Requirements:
- [TensorFlow GPU](https://www.tensorflow.org/install/gpu) requirements (optional, but recommended for proteins longer than 100 residues)
- [TensorFlow](https://github.com/tensorflow/tensorflow)
- [networkx](https://networkx.org/) 
- [logomaker](https://logomaker.readthedocs.io/en/latest/) 
- tested Python versions: 3.7.13, 3.9.2, 3.9.12
- tested TensorFlow versions: 2.5 - 2.9

1. Clone the repository: 
  <code>git clone https://github.com/Ung0d/MSA-HMM</code>
3. Install dependencies:
  <code>pip install tensorflow logomaker networkx</code>

Command line usage after manual installation:

<code>cd MSA-HMM</code>

<code>python3 MsaHmm.py -i INPUT_FILE -o OUTPUT_FILE </code>

The output directory has to exist. Currently, no other format than fasta is supported.

Interactive notebook with visualization:

Run the notebook <code>MsaHmm.ipynb</code> with juypter.

