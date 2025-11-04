Training
========

learnMSA runs a gradient-based training in order to find the best possible HMM
for aligning the sequences.
Automatically detects suitable hyperparameters for this training.
In some scenarios direct control over the training regime can be beneficial.
Possible reasons to adjust the training parameters are:

- The training fails due to memory issues or is too slow.
- The input sequences are very easy and the training should be faster.
- The time for alignment is not critical and best possible accuracy is desired.

Arguments
---------

``-n / --num_model`` *NUM_MODEL*
    This option controls how many models are trained in parallel. The models
differ slighly in their initialization, length (number of match states) and
    mini-batches seen during training.
    learnMSA automatically selects the best model according to the Akaike
    Information Criterion (AIC) after training.

    Increase this option for the potential to gain accuracy at the cost of
    longer training times and higher memory consumption.
    Reduce this option when you have limited GPU memory or want to speed up
    training.

    Default: 4

``-b / --batch`` *BATCH_SIZE*
    Controls the batch size used during training, i.e. how many sequences are
    shown to each model per training step.
    Prefer ``--tokens_per_batch`` over this option if the input sequences have
    varying lengths.
    The optimal batch size depends on the length of the input sequences and the 
    available GPU memory. Increase this value to speed up training.
    Reduce this value if you run out of GPU memory.

    Default: adaptive (typically 64–512, based on proteins and model size).

``--tokens_per_batch`` *TOKENS_PER_BATCH*
    Controls the number of tokens per batch used during training, i.e. how many
    residues are shown to each model per training step. The optimal value
    depends on the length of the input sequences and the available GPU memory.
    Increase this value to speed up training.
    Reduce this value if you run out of GPU memory.
    Prefer experimenting with this over ``--batch`` if the input sequences have
    varying lengths and if the default adaptive behavior leads to memory issues.

    Default: adaptive.

``--learning_rate`` *FLOAT*
    The learning rate used during gradient descent.

    Default: 0.05 if ``--use_language_model`` is set, otherwise 0.1.

``--epochs`` *EPOCHS [EPOCHS ...]*
    Scheme for the number of training epochs during the first, an intermediate,
    and the last iteration. Provide either a single integer (used for all
    iterations) or 3 integers (for first, intermediate, and last iteration).

    Default: [10, 2, 10]

``--max_iterations`` *MAX_ITERATIONS*
    Maximum number of training iterations. If greater than 2, model surgery
    will be applied.

    Default: 2

``--length_init_quantile`` *LENGTH_INIT_QUANTILE*
    Quantile of the input sequence lengths that defines the initial model
    lengths.

    Default: 0.5

``--surgery_quantile`` *SURGERY_QUANTILE*
    learnMSA will not use sequences shorter than this quantile for training
    during all iterations except the last.

    Default: 0.5

``--min_surgery_seqs`` *MIN_SURGERY_SEQS*
    Minimum number of sequences used per iteration. Overshadows the effect
    of ``--surgery_quantile``.

    Default: 100000

``--len_mul`` *LEN_MUL*
    Multiplicative constant for the quantile used to define the initial model
    length (see ``--length_init_quantile``).

    Default: 0.8

``--surgery_del`` *SURGERY_DEL*
    Will discard match states that are expected less often than this fraction.

    Default: 0.5

``--surgery_ins`` *SURGERY_INS*
    Will expand insertions that are expected more often than this fraction.

    Default: 0.5

``--model_criterion`` *MODEL_CRITERION*
    Criterion for model selection.
    Possible values are: `posterior`, `loglik`, `AIC`, `consensus`.

    Default: AIC

``--indexed_data``
    Don't load all data into memory at once at the cost of training time.

``--unaligned_insertions``
    Insertions will be left unaligned.

``--crop`` *CROP*
    During training, sequences longer than the given value will be cropped
    randomly. Reduces training runtime and memory usage, but might produce
    inaccurate results if too much of the sequences is cropped. The output
    alignment will not be cropped. Can be set to ``auto`` in which case
    sequences longer than 3 times the average length are cropped. Can be set
    to ``disable``.

    Default: auto

``--auto_crop_scale`` *AUTO_CROP_SCALE*
    During training sequences longer than this factor times the average length
    are cropped.

    Default: 2.0

``--frozen_insertions``
    Insertions will be frozen during training.

``--no_sequence_weights``
    Do not use sequence weights and strip mmseqs2 from requirements. In general
    not recommended.

``--skip_training``
    Skips the training phase entirely and only decodes an alignment from the provided
    model. This is useful if a pre-trained model is provided via ``--load_model``
    or the model is initialized from an existing MSA via the option ``--init_msa``.


Practical tips and example commands
-----------------------------------

Basic Usage
^^^^^^^^^^^

Standard MSA in a2m format (``--use_language_model`` is recommended but not required):

.. code-block:: bash

   learnMSA -i INPUT_FILE -o OUTPUT_FILE --use_language_model

Simple alignment without language model:

.. code-block:: bash

   learnMSA -i INPUT_FILE -o OUTPUT_FILE


Training Configuration
^^^^^^^^^^^^^^^^^^^^^^

**Quick alignment without model surgery:**

For faster results can be obtained by skipping model surgery:

.. code-block:: bash

   learnMSA -i INPUT_FILE -o OUTPUT_FILE --max_iterations 1

**High-quality alignment with more models:**

For maximum accuracy, train more models and use more iterations (requires more GPU memory and time):

.. code-block:: bash

   learnMSA -i INPUT_FILE -o OUTPUT_FILE \
       --use_language_model \
       -n 10 \
       --max_iterations 3

**Custom epoch scheme:**

Use different numbers of epochs for first, intermediate, and last iterations:

.. code-block:: bash

   learnMSA -i INPUT_FILE -o OUTPUT_FILE --epochs 20 3 20


Memory and Performance Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Limited GPU memory:**

Reduce batch size and number of models, for example:

.. code-block:: bash

   learnMSA -i INPUT_FILE -o OUTPUT_FILE -n 2 -b 32
