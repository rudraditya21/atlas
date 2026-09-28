import numpy as np

import atlas


FEATURES = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
LABELS = np.array([0, 0, 1, 1], dtype=np.uintp)


def test_train_test_and_k_fold_results_are_unpackable() -> None:
    split = atlas.train_test_split(FEATURES, LABELS, test_ratio=0.5, seed=7)
    train_features, train_labels, test_features, test_labels = split

    assert len(split) == len(train_features) + len(test_features)
    assert train_labels.shape[0] == train_features.shape[0]
    assert test_labels.shape[0] == test_features.shape[0]

    fold = atlas.k_fold_split(FEATURES, 2, seed=7)[0]
    train_indices, validation_indices = fold
    assert len(fold) == len(train_indices) + len(validation_indices)


def test_classification_report_has_mapping_protocols() -> None:
    report = atlas.classification_report(LABELS, LABELS)

    assert report["accuracy"] == report.accuracy
    assert tuple(report) == ("accuracy", "precision", "recall", "f1_score")
    assert tuple(report.keys()) == tuple(report)
    assert dict(report.items()) == dict(
        zip(report.keys(), report.values(), strict=True)
    )
    assert len(report) == 4


def test_diagnostic_results_have_value_equality() -> None:
    matrix = atlas.confusion_matrix(LABELS, LABELS)
    equivalent_matrix = atlas.confusion_matrix(LABELS, LABELS)
    report = atlas.classification_report(LABELS, LABELS)
    equivalent_report = atlas.classification_report(LABELS, LABELS)
    split = atlas.evaluate_binary_gini_split(FEATURES[:, 0], LABELS, threshold=1.5)
    equivalent_split = atlas.evaluate_binary_gini_split(
        FEATURES[:, 0], LABELS, threshold=1.5
    )

    assert matrix == equivalent_matrix
    assert report == equivalent_report
    assert split == equivalent_split
    assert matrix != report


def test_array_backed_result_metadata_matches_its_values() -> None:
    matrix = atlas.confusion_matrix(LABELS, LABELS)
    result = atlas.conjugate_gradient(
        np.eye(2, dtype=np.float64),
        np.array([1.0, 2.0], dtype=np.float64),
        max_iterations=10,
        tolerance=1e-12,
    )

    assert matrix.dtype == matrix.counts.dtype
    assert matrix.shape == matrix.counts.shape
    assert matrix.ndim == matrix.counts.ndim
    assert len(matrix) == matrix.classes.shape[0]
    assert result.dtype == result.solution.dtype
    assert result.shape == result.solution.shape
    assert result.ndim == result.solution.ndim
    assert len(result) == result.solution.shape[0]
