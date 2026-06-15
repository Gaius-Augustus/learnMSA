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
        match_variance (np.ndarray). Variances of shape `(L, d)`.
        insert_expectation (np.ndarray). Expected values of shape `(d,)`.
        insert_variance (np.ndarray). Variances of shape `(d,)`.
    """
    L: int
    match_expectations: np.ndarray
    match_variance: np.ndarray
    insert_expectation: np.ndarray
    insert_variance: np.ndarray


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

        # Initialize match_variance
        if config.match_variance is None:
            match_variance = np.zeros((L, embedding_dim), dtype=np.float32)
            match_variance += config.variance_init
        else:
            # Get variances for each match state in this head
            default_variance = np.zeros((embedding_dim,), dtype=np.float32)
            default_variance += config.variance_init
            match_variance_list = []
            for i in range(L):
                dist = get_emission_dist(
                    config.match_variance,
                    head=h,
                    index=i,
                    default=default_variance
                )
                match_variance_list.append(np.array(dist, dtype=np.float32))
            match_variance = np.stack(match_variance_list, axis=0)

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

        # Initialize insert_variance
        if config.insert_variance is None:
            insert_variance = np.zeros((embedding_dim,), dtype=np.float32)
            insert_variance += config.variance_init
        else:
            default_zeros = np.zeros((embedding_dim,), dtype=np.float32)
            default_zeros += config.variance_init
            dist = get_emission_dist(
                config.insert_variance,
                head=h,
                default=default_zeros
            )
            insert_variance = np.array(dist, dtype=np.float32)

        return cls(
            L=L,
            match_expectations=match_expectations,
            match_variance=match_variance,
            insert_expectation=insert_expectation,
            insert_variance=insert_variance,
        )
