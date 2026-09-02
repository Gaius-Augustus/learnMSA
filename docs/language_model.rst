Protein Language Model Integration
===================================

learnMSA can leverage large protein language models to generate per-token embeddings
that guide the multiple sequence alignment process. This integration can significantly
improve alignment quality, especially for distantly related sequences.


Arguments
---------

``--use_language_model``
    Uses a large protein language model to generate per-token embeddings that
    guide the MSA step. It is recommended to always use this option, unless
    computational resources are limited. Embeddings can be save to a user-
    specified file with the ``--save_emb`` option and loaded from a file with
    the ``--load_emb`` option.


``--plm_cache_dir`` *PLM_CACHE_DIR*
    Directory where the protein language model is stored.

    Default: learnMSA install directory

``--language_model`` *LANGUAGE_MODEL*
    Name of the language model to use.
    Possible values are protT5, esm2 and proteinBERT.

    Default: protT5

``--reduce_online``
    Keeps the full-dimensional embeddings and reduces them with a bottleneck
    that is trained along with the alignment, instead of projecting them once
    through the frozen scoring model. The bottleneck starts from that same
    frozen projection, so it begins where the default behaviour ends and
    adapts from there. Its reconstruction error is added to the training loss.

    .. warning::

       In this mode the embedding dataset holds the language model's full
       width -- 1024 dimensions for protT5, 64 times the default -- and it is
       held in host memory in its entirety for the whole run. **Make sure
       enough RAM is available before starting.** The embeddings are stored in
       half precision, so the requirement is about

       .. code-block::

          sum(sequence lengths) x dim x 2 bytes

       For a million residues at protT5's 1024 dimensions that is roughly
       2 GiB; for ten million residues, roughly 20 GiB. learnMSA prints the
       estimate before it starts computing.

    **PyTorch backend only.** Combining it with ``--backend tensorflow`` is an
    error.

    Default: off


Usage Example
-------------

To use protein language model integration with default settings:

.. code-block:: bash

   learnMSA -i INPUT_FILE -o OUTPUT_FILE --use_language_model

To run a different language model:

.. code-block:: bash

   learnMSA -i INPUT_FILE -o OUTPUT_FILE --use_language_model --language_model esm2

To specify a custom cache directory and language model:

.. code-block:: bash

   learnMSA -i INPUT_FILE -o OUTPUT_FILE \
       --use_language_model \
       --plm_cache_dir /path/to/cache \
       --language_model protT5

To learn the embedding reduction instead of using the frozen one:

.. code-block:: bash

   learnMSA -i INPUT_FILE -o OUTPUT_FILE --use_language_model --reduce_online


Pre-computing embeddings
------------------------

Running the language model is the expensive, one-off part of an alignment.
The ``learnMSA_embed`` script does it once and writes the embeddings to a file,
which pays off when the same sequences are aligned repeatedly, or when
embedding and aligning belong on different machines:

.. code-block:: bash

   learnMSA_embed -i INPUT_FILE -o EMBEDDING_FILE --language_model protT5

By default the embeddings are reduced by the frozen scoring model, exactly as
learnMSA would reduce them itself. Add ``--full_dim`` to keep the language
model's native width instead:

.. code-block:: bash

   learnMSA_embed -i INPUT_FILE -o EMBEDDING_FILE \
       --language_model protT5 \
       --full_dim

Either file -- or in fact any embedding file in the same format, from any
source -- is fed back with ``--load_emb``:

.. code-block:: bash

   learnMSA -i INPUT_FILE -o OUTPUT_FILE \
       --use_language_model \
       --reduce_online \
       --load_emb EMBEDDING_FILE

Reduced embeddings need ``--scoring_model_dim`` to match their width, and are
aligned without ``--reduce_online``. High-dimensional embeddings cannot be
emitted by the pHMM directly and therefore require ``--reduce_online``;
learnMSA says so explicitly rather than failing on a shape mismatch. The RAM
note above applies to reading such a file back, too.
