"""Tests for HMM configuration."""

import pytest

from learnMSA.config.hmm import HMMConfig, get_value


def test_hmm_config_scalar_initialization():
    """Test HMMConfig with scalar values for all parameters."""
    config = HMMConfig(
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
    config = HMMConfig(
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
    config = HMMConfig(
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
    config = HMMConfig(
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
    config = HMMConfig(
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


def test_hmm_config_default_values():
    """Test HMMConfig with default values."""
    config = HMMConfig()

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


def test_hmm_config_complex_multi_head():
    """Test HMMConfig with complex multi-head configuration."""
    # 3 heads with varying lengths
    config = HMMConfig(
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
    config = HMMConfig(
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
    config = HMMConfig(
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
