Initialize with existing MSA
============================

learnMSA allows to initialize the model using an existing MSA.
This can be particularly useful when you have a high-quality MSA that you want to extend with more sequences.
In contrast to other methods, learnMSA allows for a joint fine-tuning of the initial model on a large dataset.
Furthermore, you can indirectly initialize learnMSA with an existing HMM output from tools like HMMER by first converting the HMM to an MSA.

Arguments
---------

``--from_msa`` *PATH*
    If set, the initial HMM parameters will be inferred from the provided MSA in FASTA format.
    learnMSA will perform a training as usual and infer an alignment for the input
    sequences (``-i``).

    To customize the behavior of the MSA to HMM conversion, the following additional arguments are available:
    ``--match_threshold``, ``--global_factor``, ``--random_scale``, and ``--pseudocounts``.

    The training parameters can be customized to use fewer epochs (``--epochs``) and
    model surgery can be skipped (``--max_iterations 1``). Also consider reducing ``-n`` and
    ``--learning_rate``. Use ``--epochs 0 --max_iterations 1`` to disable all finetuning and
    align the input sequence directly with the HMM inferred from the MSA.

``--match_threshold`` *FLOAT*
    A value in [0, 1].
    When inferring HMM parameters from an MSA, a column is considered a match
    state if its occupancy (the fraction of non-gap characters) is at least this
    value.
    Reducing this threshold will result in a longer profile HMM with more match states.

    Default: 0.5 (50%)

``--global_factor`` *FLOAT*
    A value in [0, 1] that describes the degree to which the MSA provided
    with ``--from_msa`` is considered a global alignment. This value is used as a
    mixing factor and affects how states are counted when the data contains
    fragmentary sequences. A global alignment counts flanks as deletions,
    while a local alignment counts them as jumps into the profile using only
    a single edge.

    Default: 0.1

``--random_scale`` *FLOAT*
    The standard deviation of the Gaussian noise added to the initial parameters.
    The noise is only applied when the number of trained models
    (``-n``) is greater than 1. This can be useful if you don't trust the MSA enough,
    want to introduce some variability between the models and let learnMSA select the best one.

    Default: 0.001

``--pseudocounts``
    When set, learnMSA will add pseudocounts to the emission and transition counts
    when inferring HMM parameters from the MSA. This can help if the MSA contains few sequences
    and you don't trust its correctness. For large input MSAs this option has little effect.


Practical tips and example commands
-----------------------------------

When initializing from an existing MSA, you might want to disable model surgery, which is a heuristic aimed
to optimize the number of match states. You have a better option to adjust this number, which is ``--match_threshold``.
Use the following command for a default training without model surgery:

.. code-block:: bash

   learnMSA -i INPUT_FILE -o OUTPUT_FILE --from_msa INITIAL_MSA.fasta --max_iterations 1

When you trust the MSA strongly and want to use it with only a minor finetuning run:

.. code-block:: bash

    learnMSA -i INPUT_FILE -o OUTPUT_FILE --from_msa INITIAL_MSA.fasta --epochs 3 --max_iterations 1 --n 1


