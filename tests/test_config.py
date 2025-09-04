from pydantic import ValidationError

from learnMSA import ProfileHMMConfig


def test_states_properties():
    cases = [
        ([3],       [2*3 + 3],    [3*3 + 5],    1),
        ([3, 5],    [2*3 + 3, 2*5 + 3], [3*3 + 5, 3*5 + 5], 2),
        ([0, 2, 4], [2*0 + 3, 2*2 + 3, 2*4 + 3], [3*0 + 5, 3*2 + 5, 3*4 + 5], 3),
    ]
    for lengths, expected_states, expected_explicit, expected_heads in cases:
        cfg = ProfileHMMConfig(lengths=lengths)
        assert cfg.heads == expected_heads
        assert cfg.states == expected_states
        assert cfg.states_explicit == expected_explicit


def test_custom_probability_sequence_per_head():
    # for 2 heads
    custom = [0.2, 0.8]
    cfg = ProfileHMMConfig(lengths=[3, 5], p_begin_match=custom)
    assert isinstance(cfg.p_begin_match, list)
    assert cfg.p_begin_match == custom

    # overrides can be nested lists too
    nested = [[0.1, 0.9], [0.3, 0.7, 0.4]]
    cfg2 = ProfileHMMConfig(lengths=[2, 3], p_match_match=nested)
    assert isinstance(cfg2.p_match_match, list)
    assert cfg2.p_match_match == nested


def test_head_lists_invalid_length():
    # Test head list wrong length
    try:
        ProfileHMMConfig(
            lengths=[3, 5],
            p_begin_match=[0.2, 0.8, 0.3], # too long
            p_match_match=[0.5, 0.6],      # ok
            p_insert_insert=[0.5],         # too short
        )
        raise AssertionError(f"Expected ValidationError for {1}")
    except ValidationError as e:
        errors = e.errors()
        assert len(errors) == 2
        assert errors[0]["loc"] == ("p_begin_match",)
        assert errors[1]["loc"] == ("p_insert_insert",)


def test_nested_lists_invalid_length():
    # Test nested head list wrong length
    try:
        ProfileHMMConfig(
            lengths=[2, 3],
            p_begin_match=[[0.2, 0.8], [0.3, 0.7, 0.5]], # ok
            p_match_match=[[0.5, 0.6], [0.4, 0.5]],      # too short
            p_insert_insert=[[0.5, 0.5], [0.6, 0.6, 0.6]], # too long bc offset
        )
        raise AssertionError(f"Expected ValidationError for {1}")
    except ValidationError as e:
        errors = e.errors()
        assert len(errors) == 2
        print(errors)
        assert errors[0]["loc"] == ("p_match_match",)
        assert errors[1]["loc"] == ("p_insert_insert",)