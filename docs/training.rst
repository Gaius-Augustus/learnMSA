Training
========


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
    Reduce this option when you have limited GPU memory or want to speed up training.

    Default: 4

``-b / --batch`` *BATCH_SIZE*
    Controls the batch size used during training, i.e. how many sequences
    as shown to each model per training step.
    The optimal batch size depends on the length of the input sequences and
    the available GPU memory.
    Increase this value to speed up training.
    Reduce this value if you run out of GPU memory.

    Default: adaptive (typically 64–512, based on proteins and model size).

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

**Quick alignment with fewer iterations:**

For faster results when accuracy is less critical:

.. code-block:: bash

   learnMSA -i INPUT_FILE -o OUTPUT_FILE --max_iterations 1 --epochs 5

**High-quality alignment with more models:**

For maximum accuracy, train more models and use more iterations:

.. code-block:: bash

   learnMSA -i INPUT_FILE -o OUTPUT_FILE \
       --use_language_model \
       -n 8 \
       --max_iterations 3 \
       --epochs 15 5 15

**Custom epoch scheme:**

Use different numbers of epochs for first, intermediate, and last iterations:

.. code-block:: bash

   learnMSA -i INPUT_FILE -o OUTPUT_FILE --epochs 20 3 20


Memory and Performance Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Limited GPU memory:**

Reduce batch size and number of models:

.. code-block:: bash

   learnMSA -i INPUT_FILE -o OUTPUT_FILE -n 2 -b 32

**Large datasets:**

Use indexed data to avoid loading everything into memory:

.. code-block:: bash

   learnMSA -i INPUT_FILE -o OUTPUT_FILE --indexed_data

**Long sequences:**

Enable automatic cropping to reduce memory usage during training:

.. code-block:: bash

   learnMSA -i INPUT_FILE -o OUTPUT_FILE --crop auto --auto_crop_scale 2.5

Or set a specific crop length:

.. code-block:: bash

   learnMSA -i INPUT_FILE -o OUTPUT_FILE --crop 1000


Model Surgery Settings
^^^^^^^^^^^^^^^^^^^^^^

**Aggressive model surgery:**

Use more aggressive thresholds to create more compact models:

.. code-block:: bash

   learnMSA -i INPUT_FILE -o OUTPUT_FILE \
       --max_iterations 3 \
       --surgery_del 0.3 \
       --surgery_ins 0.7

**Conservative model surgery:**

Use less aggressive thresholds to preserve more states:

.. code-block:: bash

   learnMSA -i INPUT_FILE -o OUTPUT_FILE \
       --max_iterations 3 \
       --surgery_del 0.7 \
       --surgery_ins 0.3

**Control sequence filtering:**

Adjust which sequences are used in intermediate iterations:

.. code-block:: bash

   learnMSA -i INPUT_FILE -o OUTPUT_FILE \
       --surgery_quantile 0.3 \
       --min_surgery_seqs 50000


Advanced Options
^^^^^^^^^^^^^^^^

**Unaligned insertions:**

Keep insertions unaligned (faster, but less informative output):

.. code-block:: bash

   learnMSA -i INPUT_FILE -o OUTPUT_FILE --unaligned_insertions

**Frozen insertions:**

Freeze insertion parameters during training:

.. code-block:: bash

   learnMSA -i INPUT_FILE -o OUTPUT_FILE --frozen_insertions

**Without sequence weighting:**

Disable sequence weights (not recommended for most use cases):

.. code-block:: bash

   learnMSA -i INPUT_FILE -o OUTPUT_FILE --no_sequence_weights

**Custom model selection:**

Use BIC instead of AIC for model selection:

.. code-block:: bash

   learnMSA -i INPUT_FILE -o OUTPUT_FILE --model_criterion BIC


Tips for Specific Scenarios
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Highly diverse sequences:**

Use more iterations and a higher surgery quantile:

.. code-block:: bash

   learnMSA -i INPUT_FILE -o OUTPUT_FILE \
       --use_language_model \
       --max_iterations 4 \
       --surgery_quantile 0.7 \
       --epochs 15 3 15

**Very long sequences (>2000 residues):**

Enable cropping and reduce batch size:

.. code-block:: bash

   learnMSA -i INPUT_FILE -o OUTPUT_FILE \
       --crop auto \
       --auto_crop_scale 2.0 \
       -b 16

**Small datasets (<1000 sequences):**

Use fewer models and adjust minimum surgery sequences:

.. code-block:: bash

   learnMSA -i INPUT_FILE -o OUTPUT_FILE \
       -n 2 \
       --min_surgery_seqs 100

**Production-quality alignment:**

Recommended settings for best results:

.. code-block:: bash

   learnMSA -i INPUT_FILE -o OUTPUT_FILE \
       --use_language_model \
       -n 8 \
       --max_iterations 3 \
       --epochs 20 5 20 \
       --learning_rate 0.05
