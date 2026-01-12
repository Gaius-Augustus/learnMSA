from dataclasses import dataclass

import numpy as np

from learnMSA.msa_hmm.SequenceDataset import AlignedDataset
from learnMSA.hmm.util.transition_index_set import PHMMTransitionIndexSet
from learnMSA.config import LanguageModelConfig


@dataclass
class PHMMEmbeddingValueSet:
    """ Additional parameters for a pHMM that emits continuous embedding
    vectors. Contains the expected values and standards deviations for the
    match and insert states in each of the Z mixture components.

    Attributes:
        match_expectations (np.ndarray). Expected values of shape `(L, d)`.
        match_stddev (np.ndarray). Standard deviations of shape `(L, d)`.
        insert_expectation (np.ndarray). Expected values of shape `(d,)`.
        insert_stddev (np.ndarray). Standard deviations of shape `(d,)`.
    """
    L: int
    match_expectations: np.ndarray
    match_stddev: np.ndarray
    insert_expectation: np.ndarray
    insert_stddev: np.ndarray


    @classmethod
    def from_config(
        cls, L: int, h: int, config: LanguageModelConfig
    ) -> "PHMMEmbeddingValueSet":
        """Creates a PHMMEmbeddingValueSet from a LanguageModelConfig object.

        Args:
            length: The number of match states (L).
            h: The head index.
            config: A LanguageModelConfig object.

        Returns:
            A PHMMEmbeddingValueSet object initialized from the configuration.
        """
        # Get embedding dimension and number of mixture components
        embedding_dim = config.scoring_model_dim

        # Initialize with zeros - will be overridden by prior if needed
        match_expectations = np.zeros((L, embedding_dim), dtype=np.float32)
        match_stddev = np.random.normal(
            0.0, config.variance_init_stdev, (L, embedding_dim)
        ).astype(np.float32)
        insert_expectation = np.zeros((embedding_dim,), dtype=np.float32)
        insert_stddev = np.random.normal(
            0.0, config.variance_init_stdev, (embedding_dim,)
        ).astype(np.float32)

        return cls(
            L=L,
            match_expectations=match_expectations,
            match_stddev=match_stddev,
            insert_expectation=insert_expectation,
            insert_stddev=insert_stddev,
        )