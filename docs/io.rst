Input/output and general control
================================


Arguments
---------

``-i / --in_file`` *INPUT_FILE [INPUT_FILE ...]*
    Input fasta file containing the protein sequences to align. Any gaps
    present in the input sequences are ignored. learnMSA uses the alphabet
    ARNDCQEGHILKMFPSTWYVXUO. Special characters B, Z, J are mapped to X. The
    sequences must not contain any other non-standard characters.
    Pass several files to align each of them in a single run.

    Several input files are aligned independently of each other, but in a
    single run: learnMSA trains one profile HMM per file, all of them in
    parallel on the GPU, and writes one alignment per file. This is much
    faster than separate runs for many small families. It requires the
    pytorch backend (``--backend pytorch``), and the options that refer to a
    single dataset (``--convert``, ``--scores``, ``--decode_file``,
    ``--save_model``, ``--load_model``, ``--struct``, ``--load_emb``,
    ``--save_emb``, ``--use_language_model``, ``--from_msa``, ``--seeded``,
    ``--plot`` and ``--logo_gif``) are not available.
    Families with similar sequence lengths make the best use of the GPU.
    For example: ``learnMSA -i PF00004.fasta PF00006.fasta -o alignments/``.

``-o / --out_file`` *OUTPUT_FILE [OUTPUT_FILE ...]*
    Output file path for the resulting multiple sequence
    alignment. Use ``-f`` to change the output file type.
    LearnMSA will override existing files. Several input files need one output
    file each or a single output directory.

    With several input files, provide either one output file per input file
    (in the same order) or a single output directory. A directory receives
    one file per input file named after it, e.g. ``alignments/PF00004.a2m``
    for ``-f a2m``.

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

``--scores`` *SCORES_FILE*
    Writes per-sequence likelihoods and bit scores under the selected model to
    a file when this parameter is provided
    (e.g., ``--scores scores.tsv``). For all input sequences *s* it will report
    *log P(s)* as well as length-normalized bit scores *log P(s)/P₀(s)*, where
    *P₀* is the probability under a null model. This can be used, for example,
    to find representative sequences. Note that the raw likelihood is strongly
    correlated with sequence length. When this option is used, the ``-o / --out_file``
    parameter becomes optional.

``--compress`` *[THRESHOLD]*
    Writes the output alignment gzip-compressed and appends ``.gz`` to the
    output file name (e.g., ``-o msa.a2m --compress`` writes ``msa.a2m.gz``).
    The alignment is compressed batch by batch while it is written, so the
    uncompressed text never touches the disk. Useful when disk space is
    limited, since gap-heavy alignments of many sequences compress very well.
    If *THRESHOLD* is given, the output is only compressed if the estimated
    size of the uncompressed file exceeds *THRESHOLD* megabytes; smaller
    alignments are written as plain text. A unit suffix is accepted
    (e.g., ``--compress 500M`` or ``--compress 2G``). Only applies to the
    ``fasta`` and ``a2m`` formats; all other formats and output files are
    written uncompressed.

    Default: off. Without *THRESHOLD*, the output is always compressed.

``--struct`` *STRUCT_FILE*
    Path to a fasta file containing discrete letters from a structural alphabet
    for each sequence. Currently, only the 3Di alphabet from Foldseek is
    supported. It must be possible to match the sequences in the input fasta
    file to the sequences in the structural file by sequence ID. When this
    option is used, learnMSA will use the structural information to guide the
    alignment process.

``--load_emb`` *EMB_FILE*
    Path to a file containing embeddings for each sequence. This file should be
    a binary file in the format produced by the ``--save_emb`` option.
    When this option is used, learnMSA will load the embeddings and use them to
    guide the alignment process.

``--save_emb`` *EMB_FILE*
    Path to save computed embeddings for each sequence. Per default, stores
    embeddings in the working directory (``--work_dir``). Set to an empty
    string to disable. This option must be used together with
    ``--use_language_model`` when alignments should be computed, saved and
    used for alignment. If ``--use_language_model`` is not set, learnMSA
    will only compute and save the embeddings, but not perform any other
    computation.


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

To output sequence likelihoods for all input sequences, use the ``--scores`` option. 
This produces a tab-separated file containing log likelihoods and length-normalized 
bitscores for each sequence, which can help identify representative sequences:

.. code-block:: bash

    learnMSA -i INPUT_FILE -o OUTPUT_FILE --use_language_model --scores scores.tsv
