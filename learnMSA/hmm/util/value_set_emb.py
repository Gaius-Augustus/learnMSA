from dataclasses import dataclass

import numpy as np

from learnMSA.config import LanguageModelConfig
from learnMSA.config.util import get_emission_dist


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

        # Initialize match_expectations
        if config.match_expectations is None:
            match_expectations = np.zeros((L, embedding_dim), dtype=np.float32)
        else:
            # Get expectations for each match state in this head
            match_expectations_list = []
            default_zeros = np.zeros((embedding_dim,), dtype=np.float32)
            for i in range(L):
                dist = get_emission_dist(
                    config.match_expectations,
                    head=h,
                    index=i,
                    default=default_zeros
                )
                match_expectations_list.append(np.array(dist, dtype=np.float32))
            match_expectations = np.stack(match_expectations_list, axis=0)

        # Initialize match_stddev
        if config.match_stddev is None:
            match_stddev = np.zeros((L, embedding_dim), dtype=np.float32)
            match_stddev += config.variance_init_stdev
        else:
            # Get standard deviations for each match state in this head
            default_stddev = np.zeros((embedding_dim,), dtype=np.float32)
            default_stddev += config.variance_init_stdev
            match_stddev_list = []
            for i in range(L):
                dist = get_emission_dist(
                    config.match_stddev,
                    head=h,
                    index=i,
                    default=default_stddev
                )
                match_stddev_list.append(np.array(dist, dtype=np.float32))
            match_stddev = np.stack(match_stddev_list, axis=0)

        # Initialize insert_expectation
        if config.insert_expectation is None:
            insert_expectation = np.zeros((embedding_dim,), dtype=np.float32)
        else:
            dist = get_emission_dist(
                config.insert_expectation,
                head=h,
                default=default_zeros
            )
            insert_expectation = np.array(dist, dtype=np.float32)

        # Initialize insert_stddev
        if config.insert_stddev is None:
            insert_stddev = np.zeros((embedding_dim,), dtype=np.float32)
            insert_stddev += config.variance_init_stdev
        else:
            default_zeros = np.zeros((embedding_dim,), dtype=np.float32)
            default_zeros += config.variance_init_stdev
            dist = get_emission_dist(
                config.insert_stddev,
                head=h,
                default=default_ones
            )
            insert_stddev = np.array(dist, dtype=np.float32)

        return cls(
            L=L,
            match_expectations=match_expectations,
            match_stddev=match_stddev,
            insert_expectation=insert_expectation,
            insert_stddev=insert_stddev,
        )