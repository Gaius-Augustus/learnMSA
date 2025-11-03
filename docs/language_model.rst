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
    computational resources are limited.


``--plm_cache_dir`` *PLM_CACHE_DIR*
    Directory where the protein language model is stored.

    Default: learnMSA install directory

``--language_model`` *LANGUAGE_MODEL*
    Name of the language model to use.
    Possible values are protT5, esm2 and proteinBERT.

    Default: protT5


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
