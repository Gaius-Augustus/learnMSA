Protein Language Model Integration
===================================

learnMSA can leverage large protein language models to generate per-token embeddings
that guide the multiple sequence alignment process. This integration can significantly
improve alignment quality, especially for distant homologs or sequences with low similarity.


Arguments
---------

``--use_language_model``
    Uses a large protein language model to generate per-token embeddings that
    guide the MSA step.

    Default: False

``--plm_cache_dir`` *PLM_CACHE_DIR*
    Directory where the protein language model is stored.

    Default: learnMSA install directory

``--language_model`` *LANGUAGE_MODEL*
    Name of the language model to use.

    Default: protT5

``--scoring_model_dim`` *SCORING_MODEL_DIM*
    Reduced embedding dimension of the scoring model.

    Default: 16

``--scoring_model_activation`` *SCORING_MODEL_ACTIVATION*
    Activation function of the scoring model.

    Default: sigmoid

``--scoring_model_suffix`` *SCORING_MODEL_SUFFIX*
    Suffix to identify a specific scoring model.

    Default: (empty string)

``--temperature`` *TEMPERATURE*
    Temperature of the softmax function.

    Default: 3.0


Usage Example
-------------

To use protein language model integration with default settings:

.. code-block:: bash

   learnMSA -i INPUT_FILE -o OUTPUT_FILE --use_language_model

To specify a custom cache directory and language model:

.. code-block:: bash

   learnMSA -i INPUT_FILE -o OUTPUT_FILE \
       --use_language_model \
       --plm_cache_dir /path/to/cache \
       --language_model protT5
