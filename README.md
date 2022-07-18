# Ultra-large multiple protein alignments with a recurrent profile HMM machine learning layer

## Introduction
An aligner for potentially millions of protein sequences, based on a novel gradient descent training approach for profile Hidden Markov Models (pHMMs). We take advantage of TensorFlows automatic differentiation to differentiate the forward algorithm, i.e. compute gradients over all possible alignments.

![alt text](https://github.com/Ung0d/MSA-HMM-Analysis/blob/main/fig/boxplots_sp_homfam.png?raw=true)

Our benchmarks suggest that pHMM training is more robust with respect to scale-up in sequence numbers than traditional MSA methods. Our alignments have accuracy competitive to state of the art aligners if the number of sequences is large enough and can be done in competitive time using a GPU. If the sequence number is in the millions, we produce alignments with better accuracy and are many times faster than state of the art aligners. We think that our tool can be a first step towards a future-proof framework for ultra large MSA that lends itself to a variety of opportunities for further improvements.

***When should I use it?***

Our primary focus is on large numbers of protein sequences (starting at 10.000, the more the better) with reasonable length (at most 1.000 residues). We recommend not more than 50% of the frequences deemed fragmentary. This is a build under active development. 


## Installation

Requirements:
- [TensorFlow GPU](https://www.tensorflow.org/install/gpu) requirements (OPTIONAL, but RECOMMENDED for proteins longer than 100 residues)
- [TensorFlow](https://github.com/tensorflow/tensorflow)
- [networkx](https://networkx.org/) 
- [logomaker](https://logomaker.readthedocs.io/en/latest/) 
- tested Python versions: 3.7.13, 3.9.2, 3.9.12
- tested TensorFlow versions: 2.5, 2.7, 2.8, 2.9 [TensorFlow 2.6 is currently not supported]

1. Clone the repository: 
  <code>git clone https://github.com/Ung0d/MSA-HMM</code>
3. Install dependencies:
  <code>pip install tensorflow logomaker networkx</code>
      
## Getting Started

Command line:

<code>python3 MsaHmm.py -i path_to_input_fasta -o path_to_output_fasta </code>

<code>python3 MsaHmm.py -h</code>

The output directory has to exist. Currently, no other format than fasta is supported.

Interactive notebook with visualization:

Run the notebook <code>MsaHmm.ipynb</code> with juypter.

