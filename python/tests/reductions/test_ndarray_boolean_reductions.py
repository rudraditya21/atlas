import numpy as np

import atlas


def test_boolean_reductions_report_whole_array_truth() -> None:
    assert atlas.all(np.array([True, True], dtype=bool))
    assert not atlas.any(np.array([False, False], dtype=bool))
    assert not atlas.all(np.array([True, False], dtype=bool))
    assert atlas.any(np.array([True, False], dtype=bool))


def test_boolean_reductions_follow_logical_values_from_non_contiguous_masks() -> None:
    mask = np.array([[True, True, False], [True, True, True]], dtype=bool).T

    assert not mask.flags.c_contiguous
    assert not atlas.all(mask)
    assert atlas.any(mask)


def test_boolean_reductions_follow_empty_array_identity_values() -> None:
    values = np.array([], dtype=bool)

    assert atlas.all(values)
    assert not atlas.any(values)
