"""Tests for HMM configuration."""

from collections.abc import Sequence
from typing import cast

import pytest

from learnMSA.config import (PHMMConfig, PHMMPriorConfig, get_emission_dist,
                             get_value)


def test_hmm_config_scalar_initialization():
    """Test HMMConfig with scalar values for all parameters."""
    config = PHMMConfig(
        p_begin_match=0.5,
        p_match_match=0.7,
        p_match_insert=0.1,
        p_match_end=0.1,
        p_insert_insert=0.38,
        p_delete_delete=0.38,
        p_begin_delete=0.1,
        p_left_left=0.7,
        p_right_right=0.7,
        p_unannot_unannot=0.7,
        p_end_unannot=1e-5,
        p_end_right=0.5,
        p_start_left_flank=0.5,
    )

    assert config.p_begin_match == 0.5
    assert config.p_match_match == 0.7
    assert config.p_match_insert == 0.1
    assert config.p_match_end == 0.1
    assert config.p_insert_insert == 0.38
    assert config.p_delete_delete == 0.38
    assert config.p_begin_delete == 0.1
    assert config.p_left_left == 0.7
    assert config.p_right_right == 0.7
    assert config.p_unannot_unannot == 0.7
    assert config.p_end_unannot == 1e-5
    assert config.p_end_right == 0.5
    assert config.p_start_left_flank == 0.5


def test_hmm_config_sequence_single_head():
    """Test HMMConfig with sequence values for single head."""
    config = PHMMConfig(
        p_begin_match=[0.6],
        p_match_match=[0.75],
        p_match_insert=[0.15],
        p_match_end=[0.05],
        p_insert_insert=[0.4],
        p_delete_delete=[0.4],
        p_begin_delete=[0.2],
        p_left_left=[0.8],
        p_right_right=[0.8],
        p_unannot_unannot=[0.8],
        p_end_unannot=[1e-4],
        p_end_right=[0.6],
        p_start_left_flank=[0.4],
    )

    assert isinstance(config.p_begin_match, list)
    assert len(config.p_begin_match) == 1
    assert config.p_begin_match[0] == 0.6


def test_hmm_config_sequence_multiple_heads():
    """Test HMMConfig with sequence values for multiple heads."""
    config = PHMMConfig(
        p_begin_match=[0.5, 0.6, 0.7],
        p_match_match=[0.7, 0.75, 0.8],
        p_match_insert=[0.1, 0.15, 0.1],
        p_match_end=[0.1, 0.05, 0.05],
        p_insert_insert=[0.38, 0.4, 0.42],
        p_delete_delete=[0.38, 0.4, 0.42],
        p_begin_delete=[0.1, 0.2, 0.15],
        p_left_left=[0.7, 0.8, 0.75],
        p_right_right=[0.7, 0.8, 0.75],
        p_unannot_unannot=[0.7, 0.8, 0.75],
        p_end_unannot=[1e-5, 1e-4, 5e-5],
        p_end_right=[0.5, 0.6, 0.55],
        p_start_left_flank=[0.5, 0.4, 0.6],
    )

    assert isinstance(config.p_begin_match, list)
    assert len(config.p_begin_match) == 3
    assert config.p_begin_match[0] == 0.5
    assert config.p_begin_match[1] == 0.6
    assert config.p_begin_match[2] == 0.7


def test_hmm_config_nested_sequence():
    """Test HMMConfig with nested sequences for position-dependent parameters."""
    # For a model with 1 head and 3 match states
    config = PHMMConfig(
        p_begin_match=[[0.5, 0.3, 0.1]],  # P(Match i | Begin) for i=1,2,3
        p_match_match=[[0.7, 0.75, 0.8]],  # P(Match i+1 | Match i) for i=1,2,3
        p_match_insert=[[0.1, 0.15]],  # P(Insert i | Match i) for i=1,2
        p_match_end=[[0.05, 0.05]],  # P(End | Match i) for i=1,2
        p_insert_insert=[[0.4, 0.35]],  # P(Insert i | Insert i) for i=1,2
        p_delete_delete=[[0.4, 0.35]],  # P(Delete i+1 | Delete i) for i=1,2
    )

    assert isinstance(config.p_begin_match, list)
    assert len(config.p_begin_match) == 1
    assert isinstance(config.p_begin_match[0], list)
    assert len(config.p_begin_match[0]) == 3
    assert config.p_begin_match[0][0] == 0.5
    assert config.p_begin_match[0][1] == 0.3
    assert config.p_begin_match[0][2] == 0.1


def test_hmm_config_mixed_scalar_and_sequence():
    """Test HMMConfig with mixed scalar and sequence parameters."""
    config = PHMMConfig(
        p_begin_match=[0.5, 0.6],  # Sequence for 2 heads
        p_match_match=0.7,  # Scalar (same for all)
        p_match_insert=[0.1, 0.15],  # Sequence for 2 heads
        p_match_end=0.1,  # Scalar
        p_insert_insert=0.38,  # Scalar
        p_delete_delete=0.38,  # Scalar
    )

    assert isinstance(config.p_begin_match, list)
    assert config.p_match_match == 0.7
    assert isinstance(config.p_match_insert, list)


def test_get_value_scalar():
    """Test get_value function with scalar parameters."""
    assert get_value(0.5, head=0) == 0.5
    assert get_value(0.5, head=5) == 0.5
    assert get_value(0.7, head=0, index=0) == 0.7


def test_get_value_sequence_of_floats():
    """Test get_value function with sequence of floats."""
    param = [0.5, 0.6, 0.7]
    assert get_value(param, head=0) == 0.5
    assert get_value(param, head=1) == 0.6
    assert get_value(param, head=2) == 0.7


def test_get_value_nested_sequence():
    """Test get_value function with nested sequences."""
    param = [[0.5, 0.3, 0.2], [0.6, 0.25, 0.15]]

    assert get_value(param, head=0, index=0) == 0.5
    assert get_value(param, head=0, index=1) == 0.3
    assert get_value(param, head=0, index=2) == 0.2
    assert get_value(param, head=1, index=0) == 0.6
    assert get_value(param, head=1, index=1) == 0.25
    assert get_value(param, head=1, index=2) == 0.15


def test_get_value_nested_sequence_requires_index():
    """Test that get_value raises error when index is missing for nested sequences."""
    param = [[0.5, 0.3, 0.2]]

    with pytest.raises(ValueError, match="Index must be provided"):
        get_value(param, head=0)


def test_get_value_invalid_parameter():
    """Test get_value with invalid parameter types."""
    with pytest.raises(ValueError, match="Invalid parameter type"):
        get_value("invalid", head=0)

    with pytest.raises(ValueError, match="Invalid parameter type"):
        get_value([["nested", "strings"]], head=0, index=0)


def test_hmm_config_complex_multi_head():
    """Test HMMConfig with complex multi-head configuration."""
    # 3 heads with varying lengths
    config = PHMMConfig(
        # Head-specific scalar values
        p_begin_match=[0.5, 0.55, 0.6],
        p_match_match=[0.7, 0.75, 0.8],
        p_match_insert=[0.1, 0.12, 0.09],
        p_match_end=[0.1, 0.08, 0.07],
        # Position-dependent values for one head
        p_insert_insert=[0.38, 0.4, 0.42],
        p_delete_delete=[0.38, 0.4, 0.42],
        # Flank parameters
        p_left_left=[0.7, 0.75, 0.8],
        p_right_right=[0.7, 0.75, 0.8],
        p_unannot_unannot=[0.7, 0.75, 0.8],
        p_end_unannot=[1e-5, 2e-5, 3e-5],
        p_end_right=[0.5, 0.55, 0.6],
        p_start_left_flank=[0.5, 0.45, 0.55],
    )

    # Test head 0
    assert get_value(config.p_begin_match, 0) == 0.5
    assert get_value(config.p_match_match, 0) == 0.7
    assert get_value(config.p_left_left, 0) == 0.7

    # Test head 1
    assert get_value(config.p_begin_match, 1) == 0.55
    assert get_value(config.p_match_match, 1) == 0.75
    assert get_value(config.p_left_left, 1) == 0.75

    # Test head 2
    assert get_value(config.p_begin_match, 2) == 0.6
    assert get_value(config.p_match_match, 2) == 0.8
    assert get_value(config.p_left_left, 2) == 0.8


def test_hmm_config_position_dependent_multi_head():
    """Test HMMConfig with position-dependent parameters across multiple heads."""
    # 2 heads with position-dependent match transitions
    config = PHMMConfig(
        p_begin_match=[
            [0.5, 0.3, 0.2],  # Head 0: P(M1|B), P(M2|B), P(M3|B)
            [0.6, 0.25, 0.15]  # Head 1: P(M1|B), P(M2|B), P(M3|B)
        ],
        p_match_match=[
            [0.7, 0.75, 0.8],  # Head 0: P(M2|M1), P(M3|M2), P(M4|M3)
            [0.72, 0.78, 0.82]  # Head 1
        ],
        p_match_insert=[
            [0.1, 0.12],  # Head 0: P(I1|M1), P(I2|M2)
            [0.11, 0.10]  # Head 1
        ],
    )

    # Test head 0, position 0
    assert get_value(config.p_begin_match, 0, 0) == 0.5
    assert get_value(config.p_match_match, 0, 0) == 0.7
    assert get_value(config.p_match_insert, 0, 0) == 0.1

    # Test head 0, position 1
    assert get_value(config.p_begin_match, 0, 1) == 0.3
    assert get_value(config.p_match_match, 0, 1) == 0.75
    assert get_value(config.p_match_insert, 0, 1) == 0.12

    # Test head 1, position 0
    assert get_value(config.p_begin_match, 1, 0) == 0.6
    assert get_value(config.p_match_match, 1, 0) == 0.72
    assert get_value(config.p_match_insert, 1, 0) == 0.11


def test_hmm_config_edge_cases():
    """Test HMMConfig with edge case values."""
    config = PHMMConfig(
        p_begin_match=0.0,  # Minimum probability
        p_match_match=1.0,  # Maximum probability
        p_end_unannot=0.0,  # Zero probability
    )

    assert config.p_begin_match == 0.0
    assert config.p_match_match == 1.0
    assert config.p_end_unannot == 0.0

    # Test get_value with these edge cases
    assert get_value(config.p_begin_match, 0) == 0.0
    assert get_value(config.p_match_match, 0) == 1.0
    assert get_value(config.p_end_unannot, 0) == 0.0


# Tests for emission parameters

def test_hmm_config_default_emissions():
    """Test HMMConfig with default emission parameters."""
    config = PHMMConfig()

    assert config.match_emissions is None
    assert config.insert_emissions is None
    assert len(config.background_distribution) == 23  # Default alphabet size


def test_hmm_config_match_emissions_single_distribution():
    """Test HMMConfig with single emission distribution for all match states."""
    # Single distribution applied to all match states in all heads
    dist = [0.05] * 23
    config = PHMMConfig(
        match_emissions=dist,
        use_prior_for_emission_init=False,
    )

    assert config.match_emissions is not None
    assert len(config.match_emissions) == 23


def test_hmm_config_match_emissions_head_specific():
    """Test HMMConfig with head-specific match emission distributions."""
    # Different distribution per head, same for all positions within head
    dist1 = [0.05] * 23
    dist2 = [0.04] * 23
    config = PHMMConfig(
        match_emissions=[dist1, dist2],
        use_prior_for_emission_init=False,
    )

    assert config.match_emissions is not None
    assert isinstance(config.match_emissions, Sequence)
    # Use casts to satisfy type checker
    match_emissions = cast(Sequence, config.match_emissions)
    assert len(match_emissions) == 2
    assert len(cast(Sequence, match_emissions[0])) == 23
    assert len(cast(Sequence, match_emissions[1])) == 23


def test_hmm_config_match_emissions_fully_specified():
    """Test HMMConfig with fully specified position-dependent emissions."""
    # Different distribution per position in each head
    # Head 0: 3 match states
    dist_h0_m1 = [0.05] * 23
    dist_h0_m2 = [0.04] * 23
    dist_h0_m3 = [0.06] * 23
    # Head 1: 2 match states
    dist_h1_m1 = [0.045] * 23
    dist_h1_m2 = [0.055] * 23

    config = PHMMConfig(
        match_emissions=[
            [dist_h0_m1, dist_h0_m2, dist_h0_m3],
            [dist_h1_m1, dist_h1_m2]
        ],
        use_prior_for_emission_init=False,
    )

    assert config.match_emissions is not None
    assert isinstance(config.match_emissions, Sequence)
    match_emissions = cast(Sequence, config.match_emissions)
    assert len(match_emissions) == 2
    assert len(cast(Sequence, match_emissions[0])) == 3
    assert len(cast(Sequence, match_emissions[1])) == 2
    assert len(cast(Sequence, cast(Sequence, match_emissions[0])[0])) == 23


def test_hmm_config_insert_emissions_single_distribution():
    """Test HMMConfig with single insertion emission distribution."""
    dist = [0.05] * 23
    config = PHMMConfig(insert_emissions=dist)

    assert config.insert_emissions is not None
    assert len(config.insert_emissions) == 23


def test_hmm_config_insert_emissions_head_specific():
    """Test HMMConfig with head-specific insertion emissions."""
    dist1 = [0.05] * 23
    dist2 = [0.04] * 23
    dist3 = [0.06] * 23
    config = PHMMConfig(insert_emissions=[dist1, dist2, dist3])

    assert config.insert_emissions is not None
    assert isinstance(config.insert_emissions, Sequence)
    insert_emissions = cast(Sequence, config.insert_emissions)
    assert len(insert_emissions) == 3
    assert len(cast(Sequence, insert_emissions[0])) == 23
    assert len(cast(Sequence, insert_emissions[1])) == 23
    assert len(cast(Sequence, insert_emissions[2])) == 23


def test_hmm_config_emissions_validation_wrong_alphabet_size():
    """Test that emissions with wrong alphabet size are rejected."""
    dist_wrong_size = [0.05] * 20  # Should be 23

    with pytest.raises(ValueError, match="alphabet size"):
        PHMMConfig(match_emissions=dist_wrong_size)

    with pytest.raises(ValueError, match="alphabet size"):
        PHMMConfig(insert_emissions=dist_wrong_size)


def test_hmm_config_emissions_with_custom_alphabet():
    """Test emissions with custom alphabet."""
    custom_alphabet = "ACGT"
    dist = [0.25] * 4

    config = PHMMConfig(
        alphabet=custom_alphabet,
        match_emissions=dist,
        insert_emissions=dist,
        use_prior_for_emission_init=False,
    )

    assert config.alphabet == custom_alphabet
    assert isinstance(config.match_emissions, Sequence)
    assert isinstance(config.insert_emissions, Sequence)
    assert len(config.match_emissions) == 4
    assert len(config.insert_emissions) == 4


def test_get_emission_dist_with_none():
    """Test get_emission_dist returns default when param is None."""
    default = [0.05] * 23
    result = get_emission_dist(None, head=0, default=default)
    assert result == default


def test_get_emission_dist_with_none_no_default():
    """Test get_emission_dist raises error when no default provided."""
    with pytest.raises(ValueError, match="No emission distribution"):
        get_emission_dist(None, head=0)


def test_get_emission_dist_single_distribution():
    """Test get_emission_dist with single distribution for all."""
    dist = [0.05] * 23

    # Should return same distribution regardless of head
    assert get_emission_dist(dist, head=0) == dist
    assert get_emission_dist(dist, head=1) == dist
    assert get_emission_dist(dist, head=5) == dist


def test_get_emission_dist_head_specific():
    """Test get_emission_dist with head-specific distributions."""
    dist1 = [0.05] * 23
    dist2 = [0.04] * 23
    dist3 = [0.06] * 23
    param = [dist1, dist2, dist3]

    assert get_emission_dist(param, head=0) == dist1
    assert get_emission_dist(param, head=1) == dist2
    assert get_emission_dist(param, head=2) == dist3


def test_get_emission_dist_position_specific():
    """Test get_emission_dist with position-specific distributions."""
    dist_h0_m0 = [0.05] * 23
    dist_h0_m1 = [0.04] * 23
    dist_h1_m0 = [0.06] * 23
    dist_h1_m1 = [0.045] * 23

    param = [
        [dist_h0_m0, dist_h0_m1],
        [dist_h1_m0, dist_h1_m1]
    ]

    assert get_emission_dist(param, head=0, index=0) == dist_h0_m0
    assert get_emission_dist(param, head=0, index=1) == dist_h0_m1
    assert get_emission_dist(param, head=1, index=0) == dist_h1_m0
    assert get_emission_dist(param, head=1, index=1) == dist_h1_m1


def test_get_emission_dist_position_specific_requires_index():
    """Test that get_emission_dist raises error when index missing."""
    param = [
        [[0.05] * 23, [0.04] * 23],
    ]

    with pytest.raises(ValueError, match="Index must be provided"):
        get_emission_dist(param, head=0)


def test_hmm_config_combined_emissions_and_transitions():
    """Test HMMConfig with both custom emissions and transitions."""
    match_dist = [0.05] * 23
    insert_dist = [0.04] * 23

    config = PHMMConfig(
        match_emissions=match_dist,
        insert_emissions=insert_dist,
        p_begin_match=0.6,
        p_match_match=0.75,
        p_match_insert=0.15,
        use_prior_for_emission_init=False,
    )

    assert config.match_emissions == match_dist
    assert config.insert_emissions == insert_dist
    assert config.p_begin_match == 0.6
    assert config.use_prior_for_emission_init is False


def test_hmm_config_use_prior_flag():
    """Test use_prior_for_emission_init flag behavior."""
    config_default = PHMMConfig()
    assert config_default.use_prior_for_emission_init is True

    config_explicit = PHMMConfig(use_prior_for_emission_init=False)
    assert config_explicit.use_prior_for_emission_init is False

    # When use_prior is False, custom emissions can still be None
    # (will use background_distribution)
    config_with_emissions = PHMMConfig(
        use_prior_for_emission_init=False,
        match_emissions=None,
        insert_emissions=None
    )
    assert config_with_emissions.match_emissions is None
    assert config_with_emissions.insert_emissions is None


def test_hmm_prior_config_alpha_defaults():
    """Test HMMPriorConfig alpha parameter default values."""
    config = PHMMPriorConfig()
    assert config.alpha_flank == 7000.0
    assert config.alpha_single == 1e9
    assert config.alpha_global == 1e4
    assert config.alpha_flank_compl == 1.0
    assert config.alpha_single_compl == 1e-3
    assert config.alpha_global_compl == 1.0
    assert config.epsilon == 1e-16


def test_hmm_prior_config_alpha_custom_values():
    """Test HMMPriorConfig with custom alpha parameter values."""
    config = PHMMPriorConfig(
        alpha_flank=5000.0,
        alpha_single=1e8,
        alpha_global=5000.0,
        alpha_flank_compl=2.0,
        alpha_single_compl=3.0,
        alpha_global_compl=4.0,
        epsilon=1e-12,
    )
    assert config.alpha_flank == 5000.0
    assert config.alpha_single == 1e8
    assert config.alpha_global == 5000.0
    assert config.alpha_flank_compl == 2.0
    assert config.alpha_single_compl == 3.0
    assert config.alpha_global_compl == 4.0
    assert config.epsilon == 1e-12


def test_alpha_flank_validation():
    """Test that alpha_flank must be positive."""
    # Valid values
    PHMMPriorConfig(alpha_flank=0.1)
    PHMMPriorConfig(alpha_flank=10000)

    # Invalid values
    with pytest.raises(
        ValueError,
        match="alpha_flank must be greater than 0"
    ):
        PHMMPriorConfig(alpha_flank=0)

    with pytest.raises(
        ValueError,
        match="alpha_flank must be greater than 0"
    ):
        PHMMPriorConfig(alpha_flank=-100)


def test_alpha_single_validation():
    """Test that alpha_single must be positive."""
    # Valid values
    PHMMPriorConfig(alpha_single=1)
    PHMMPriorConfig(alpha_single=1e10)

    # Invalid values
    with pytest.raises(
        ValueError,
        match="alpha_single must be greater than 0"
    ):
        PHMMPriorConfig(alpha_single=0)

    with pytest.raises(
        ValueError,
        match="alpha_single must be greater than 0"
    ):
        PHMMPriorConfig(alpha_single=-1)


def test_alpha_global_validation():
    """Test that alpha_global must be positive."""
    # Valid values
    PHMMPriorConfig(alpha_global=100)
    PHMMPriorConfig(alpha_global=1e6)

    # Invalid values
    with pytest.raises(
        ValueError,
        match="alpha_global must be greater than 0"
    ):
        PHMMPriorConfig(alpha_global=0)

    with pytest.raises(
        ValueError,
        match="alpha_global must be greater than 0"
    ):
        PHMMPriorConfig(alpha_global=-500)


def test_alpha_complement_validation():
    """Test that alpha complement parameters must be positive."""
    # Valid values
    PHMMPriorConfig(alpha_flank_compl=0.5)
    PHMMPriorConfig(alpha_single_compl=10)
    PHMMPriorConfig(alpha_global_compl=100)

    # Invalid values
    with pytest.raises(
        ValueError,
        match="alpha_flank_compl must be greater than 0"
    ):
        PHMMPriorConfig(alpha_flank_compl=0)

    with pytest.raises(
        ValueError,
        match="alpha_single_compl must be greater than 0"
    ):
        PHMMPriorConfig(alpha_single_compl=-1)

    with pytest.raises(
        ValueError,
        match="alpha_global_compl must be greater than 0"
    ):
        PHMMPriorConfig(alpha_global_compl=-5)


def test_epsilon_validation():
    """Test that epsilon must be positive."""
    # Valid values
    PHMMPriorConfig(epsilon=1e-20)
    PHMMPriorConfig(epsilon=0.1)

    # Invalid values
    with pytest.raises(
        ValueError,
        match="epsilon must be greater than 0"
    ):
        PHMMPriorConfig(epsilon=0)

    with pytest.raises(
        ValueError,
        match="epsilon must be greater than 0"
    ):
        PHMMPriorConfig(epsilon=-1e-10)

