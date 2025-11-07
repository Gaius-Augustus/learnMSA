API Reference
=============

learnMSA: Learning and Aligning Large Protein Families with support of protein language models.


Sequence Dataset
----------------

Classes for managing sequence data (either unaligned or aligned).

.. autoclass:: learnMSA.msa_hmm.SequenceDataset.SequenceDataset
.. autoclass:: learnMSA.msa_hmm.SequenceDataset.AlignedDataset


Configuration
-------------

Configuration classes for learnMSA.

.. autoclass:: learnMSA.Configuration
.. autoclass:: learnMSA.config.training.TrainingConfig
.. autoclass:: learnMSA.config.input_output.InputOutputConfig
.. autoclass:: learnMSA.config.language_model.LanguageModelConfig
.. autoclass:: learnMSA.config.init_msa.InitMSAConfig
.. autoclass:: learnMSA.config.visualization.VisualizationConfig
.. autoclass:: learnMSA.config.advanced.AdvancedConfig