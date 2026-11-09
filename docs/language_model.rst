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

``--plm_cluster_id`` *PLM_CLUSTER_ID*
    Sequence identity at which sequences are clustered before the language
    model runs. The language model runs only on the cluster representatives.
    A member joins a cluster if its alignment to the representative has at
    least this sequence identity and covers 95% of both sequences.
    Cluster members copy embedding vectors at positions aligned to their
    representative.
    Also applies to the 3Di logits that ProstT5 predicts with ``--struct``.
    1.0 only clusters identical sequences, which gives the same
    result faster. Values below 1 save more runtime and require mmseqs2.

    Default: 1.0


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
