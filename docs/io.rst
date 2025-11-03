Input/output and general control
================================


Arguments
---------

``-i / --in_file`` *INPUT_FILE*
    Input fasta file containing the protein sequences to align. Any gaps
    present in the input sequences are ignored. learnMSA uses the alphabet
    ARNDCQEGHILKMFPSTWYVXUO. Special characters B, Z, J are mapped to X. The
    sequences must not contain any other non-standard characters.

``-o / --out_file`` *OUTPUT_FILE*
    Output file path for the resulting multiple sequence
    alignment. Use ``-f`` to change the output file type.
    LearnMSA will override existing files.

``-f / --format`` *FORMAT*
    Format of the output alignment file.
    Per default, learnMSA outputs alignments in a2m format.
    This format is closely related to fasta and usually compatible with fasta parsers.
    In addition to fasta, a2m uses lower case letters to indicate insertions
    with respect to the profile HMM and uses dots (.) to represent an insertion
    in other sequences at the same position. It uses upper case letters for match states
    and dashes (-) for deletions.
    The format can be set to "fasta", which uses only standard dashes and upper
    case letters. Use ``--convert`` to quickly convert between different formats.
    The a2m and fasta options use a maximum line length of 80 characters.

    Furthermore, this option be set to any valid Biopython SeqIO format, in which
    case learnMSA will write a fasta file and automatically converts it.
    This is not recommended for large alignments, as output files can be very
    large and the file contents can not be streamed.

    Default: a2m (fasta).

``--convert`` *MSA_FILE*
    With this option, learnMSA does not perform any alignment, but
    only converts the input MSA to the format specified with ``-f``.
    For example, to convert an a2m file to fasta format, use:
    ``learnMSA -i proteins.a2m --convert -f fasta -o protein.fasta``.

``-s / --silent``
    Suppresses all standard output messages.

``-d / --cuda_visible_devices``
    Controls the GPU devices visible to learnMSA as a comma-
    separated list of device IDs. The value -1 forces learnMSA
    to run on CPU. Per default, learnMSA attempts to use all
    available GPUs. Use ``-d i`` to use a specific GPU, where i is the GPU ID starting from 0.

``--work_dir`` *WORK_DIR*
    Directory where any secondary files are stored.

    Default: ./tmp

``--save_model`` *MODEL_FILE*
    If set, the trained model parameters will be saved to the specified file.
    The file format is meant to be read with the ``--load_model`` option only.

``--load_model`` *MODEL_FILE*
    If set, learnMSA will load the model parameters from the specified file
    and use them as initialization for training. Use the ``--skip_training`` option
    to directly align the input sequences without further training.

Practical tips and example commands
-----------------------------------

Standard MSA in a2m format (``--use_language_model`` is recommended but not required):

.. code-block:: bash

   learnMSA -i INPUT_FILE -o OUTPUT_FILE --use_language_model

Enforce fasta output if a2m leads to compatibility issues:

.. code-block:: bash

    learnMSA -i INPUT_FILE -o OUTPUT_FILE --use_language_model -f fasta

To control where learnMSA writes temporary files, use the ``--work_dir`` option.
In particular, this is useful when aligning the same input file multiple times
in parallel, to avoid conflicts between different runs:

.. code-block:: bash

    learnMSA -i INPUT_FILE -o OUTPUT_FILE --use_language_model --work_dir ./my_temp_dir

To save a trained model for later reuse, use the ``--save_model`` option:

.. code-block:: bash

    learnMSA -i INPUT_FILE -o OUTPUT_FILE --use_language_model --save_model my_model

This can be useful to reproduce alignments later on or to resume a training.

To load a previously saved model, use the ``--load_model`` option. You may combine it with ``--skip_training`` to directly align
the input sequences without further training:

.. code-block:: bash

    learnMSA -i INPUT_FILE -o OUTPUT_FILE --load_model my_model --skip_training
