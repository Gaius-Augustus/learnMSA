# Ultra-large multiple protein alignments with a recurrent profile HMM machine learning layer

## Introduction
An aligner for potentially millions of protein sequences, based on a novel gradient descent training approach for profile Hidden Markov Models (pHMMs). We take advantage of TensorFlows automatic differentiation to compute gradients of the likelihood over all possible alignments.

Our benchmarks suggest that pHMM training is more robust with respect to scale-up in sequence numbers than traditional MSA methods. Our alignments have accuracy competitive to state of the art aligners if the number of sequences is large enough and can be done in competitive time using a GPU. If the sequence number is the millions, we produce alignments with better accuracy and are many times faster than state of the art aligners. We think that our tool can be a first step towards a future-proof framework for ultra large MSA that lends itself to a variety of opportunities for further improvements.

## Installation

Requirements:
- python 3.x (todo: version test)
- [TensorFlow](https://github.com/tensorflow/tensorflow)
- [GPU](https://www.tensorflow.org/install/gpu) requirements of tensorflow (OPTIONAL, but RECOMMENDED for proteins longer than 100 residues)
- [networkx](https://networkx.org/) (OPTIONAL, only for pHMM visualization)
- [logomaker](https://logomaker.readthedocs.io/en/latest/) (OPTIONAL, only for pHMM visualization)

1. Clone the repository: 
  <code>git clone https://github.com/Ung0d/MSA-HMM</code>
3. Install dependencies 

      Just alignments: <code>pip install tensorflow==2.5.0</code>

      With pHMM visualization: <code>pip install tensorflow==2.5.0 logomaker networkx</code>
