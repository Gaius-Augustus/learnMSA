*Our tool is under active development and feedback is very much appreciated.*

# learnMSA2: deep protein multiple alignments with large language and hidden Markov models

<img src="https://github.com/Gaius-Augustus/learnMSA/blob/main/logo/training_loop.gif" alt="" loop=infinite>

## Features

- Aligns large numbers of protein sequences with above state-of-the-art accuracy
- Can utilize protein language models (`--use_language_model`) for significantly improved accuracy
- Enables ultra-large alignment of millions of sequences 
- GPU acceleration, multi-GPU support
- Scales linear in the number of sequences (does not require a guide tree)
- Memory efficient (depending on sequence length, aligning millions of sequences on a laptop is possible)
- Visualize a profile HMM or a sequence logo of the consensus motif

## Current limitations

- Requires many sequences (in most cases starting at 1000, a few 100 might still be enough) to achieve high accuracy
- Only for protein sequences
- Increasingly slow for long proteins with a length > 1000 residues

# Installation

*Recommended ways to install learnMSA along with Tensorflow + GPU:*

## Singularity/Docker

We provide a hassle-free docker image including GPU and pLM support.

```
singularity build learnmsa.sif docker://felbecker/learnmsa
singularity run --nv learnmsa.sif learnMSA
```

Running the container with `--nv` is required for GPU support.

## conda/mamba and pip

1. Create a conda environment:

```
conda create -n learnMSA python=3.12
conda activate learnMSA
```

2. Install learnMSA (cuda toolkit included):

```
pip install learnMSA
```

3. Additional installs for sequence weights (recommended!):
   
```
conda install -c bioconda mmseqs2
```

You may have to set up [Bioconda channels](https://bioconda.github.io/).

4. Additional installs for language model support (recommended!):
   
```
pip install torch==2.2.1 tf-keras==2.17.0
```

5. (optional) Verify that TensorFlow and learnMSA are correctly installed:

```
python3 -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
learnMSA -h
```

## bioconda (currently not recommended)

`conda install -c bioconda -n learnMSA learnMSA`

The additional installs (steps 3. and 4. above) are again highly recommended.

Unfortunately, there are errors to be expected when installing learnMSA with bioconda.

### ${\text{\color{red}Troubleshooting:}}$

### Error:
`tensorflow.python.framework.errors_impl.UnknownError: {{function_node __wrapped__Expm1_device_/job:localhost/replica:0/task:0/device:GPU:0}} JIT compilation failed.`

Your root error is:
`TensorFlow libdevice not found`

Fix:

Find `nvvm` directory:
`find / -type d -name nvvm 2>/dev/null`

Expected outputs:
`<path>/nvvm`

If there are multiple paths, choose the one matching your conda environment.

Run:
`export XLA_FLAGS=--xla_gpu_cuda_data_dir=<path>`


### Error:
`ERROR: Flag 'minloglevel' was defined more than once (...)`

Fix:
`pip install --no-deps --upgrade sentencepiece==0.1.99`



## Using learnMSA for alignment

Recommended way to align proteins **with learnMSA version >= 2.0.10**:

<code>learnMSA -i INPUT_FILE -o OUTPUT_FILE --use_language_model</code>


Recommended way to align proteins **with learnMSA version < 2.0.10**:

<code>learnMSA -i INPUT_FILE -o OUTPUT_FILE --use_language_model --sequence_weights</code>

Without language model support (recommended for speed and for proteins with very high sequence similarity):

<code>learnMSA -i INPUT_FILE -o OUTPUT_FILE</code>

Note: If you installed learnMSA via docker/singularity, you have to run `singularity run --nv learnmsa.sif learnMSA -i ...`

Note 2: Since v2.0.10 **the default behavior changed**: Sequence weights are now used by default and the `--sequence_weights` option was removed. Instead, a `--no_sequence_weights` option exists to align without sequence weights (not recommended). Users that installed learnMSA via pip have to manually install mmseqs2 (conda is recommended, see above).

To output a pdf with a sequence logo alongside the msa, use `--logo`. For a fun gif that visualizes the training process, you can use `--logo_gif` (attention, slows down training and should not be used for real alignments).
  
## Interactive notebook with visualization:

Run the notebooks <code>learnMSA_demo.ipynb</code> or <code>learnMSA_with_language_model_demo.ipynb</code> with juypter.
  
## Benchmark:

![alt text](https://github.com/felbecker/snakeMSA/blob/main/plots/barplots.png?raw=true)

# Publications

Becker F, Stanke M. **learnMSA2: deep protein multiple alignments with large language and hidden Markov models**. *Bioinformatics*. 2024

Becker F, Stanke M. **learnMSA: learning and aligning large protein families**. *GigaScience*. 2022
