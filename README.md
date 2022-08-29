# learnMSA: Learning and Aligning large Protein Families

## Introduction
Multiple sequence alignment formulated as a statistical machine learning problem, where an optimal profile hidden Markov model for a potentially very large family of protein sequences is searched and an alignment is decoded. We use an automatically differentiable variant of the Forward algorithm.

## Installation

#### Using Bioconda:
  
  If you haven't done it yet, set up [Bioconda channels](https://bioconda.github.io/) first.
  
  We recommend to install in a clean environment:
  
  <code>conda create -n learnMSA learnMSA</code>
  
  which creates an environment called learnMSA and installs the package learnMSA.
  
  To run learnMSA, you have to activate the environment first:
  
  <code>conda activate learnMSA</code>.
  
  If you do not want to use environments, we recommend to use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to have a clearn base environment.

#### Using pip:

  <code>pip install learnMSA</code>
  
#### Command line use after installing with Bioconda or pip:

<code>learnMSA -i INPUT_FILE -o OUTPUT_FILE</code>
  
<code>learnMSA -h</code>

#### Manual installation:

Requirements:
- meet the [TensorFlow GPU](https://www.tensorflow.org/install/gpu) requirements (optional, but recommended for proteins longer than 100 residues)
- [TensorFlow](https://github.com/tensorflow/tensorflow) (tested versions: 2.5, >=2.7)
- [networkx](https://networkx.org/) 
- [logomaker](https://logomaker.readthedocs.io/en/latest/) 
- python >= 3.7

1. Clone the repository: 

  <code>git clone https://github.com/Gaius-Augustus/learnMSA</code>
  
2. Install dependencies:

  <code>pip install tensorflow logomaker networkx</code>
  
3. Run:

  <code>cd learnMSA</code>
  
  <code>python3 learnMSA.py --help</code>
  

#### Interactive notebook with visualization:

Run the notebook <code>MsaHmm.ipynb</code> with juypter.

