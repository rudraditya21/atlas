import copy

import numpy as np
import pytest

import atlas


FEATURES = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
TARGETS = np.array([0.0, 1.0, 2.0], dtype=np.float64)
LABELS = np.array([0, 0, 1], dtype=np.uintp)


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (lambda: atlas.RidgeRegression(0.5), {"l2_regularization": 0.5}),
        (
            lambda: atlas.BinaryLogisticRegression(max_iterations=20),
            {
                "learning_rate": 0.1,
                "max_iterations": 20,
                "convergence_tolerance": 1e-6,
                "l2_regularization": 0.0,
            },
        ),
        (
            lambda: atlas.BinaryPerceptron(max_iterations=20, shuffle_seed=7),
            {"learning_rate": 0.1, "max_iterations": 20, "shuffle_seed": 7},
        ),
        (
            lambda: atlas.GaussianNaiveBayes(1e-6),
            {"variance_smoothing": 1e-6},
        ),
        (
            lambda: atlas.KnnClassifier(1, search_algorithm="auto"),
            {
                "k": 1,
                "search_algorithm": "auto",
                "weighting": "uniform",
                "tree_leaf_size": 1,
            },
        ),
        (
            lambda: atlas.KnnRegressor(1, weighting="distance"),
            {
                "k": 1,
                "search_algorithm": "brute_force",
                "weighting": "distance",
                "tree_leaf_size": 1,
            },
        ),
        (
            lambda: atlas.MinMaxScaler(-1.0, 1.0),
            {"output_minimum": -1.0, "output_maximum": 1.0},
        ),
    ],
)
def test_configurable_models_expose_stable_parameters_and_unfitted_copies(
    factory, expected: dict[str, object]
) -> None:
    model = factory()

    assert not model.is_fitted
    assert model.get_params() == expected
    assert repr(model).startswith(type(model).__name__)

    copied = model.copy()
    assert copied.get_params() == expected
    assert not copied.is_fitted
    assert copy.copy(model).get_params() == expected


def test_linear_model_fitted_state_and_learned_parameters() -> None:
    model = atlas.RidgeRegression(0.5)

    model.fit(FEATURES, TARGETS)

    assert model.is_fitted
    assert isinstance(model.intercept_, float)
    assert model.coef_.shape == (1,)

    model.set_params(l2_regularization=1.0)
    assert not model.is_fitted


def test_scaler_fitted_state_parameters_and_copying() -> None:
    scaler = atlas.MinMaxScaler(-1.0, 1.0)

    transformed = scaler.fit_transform(FEATURES)

    assert scaler.is_fitted
    assert scaler.data_min_.shape == (1,)
    assert scaler.data_max_.shape == (1,)
    assert transformed.shape == FEATURES.shape
    assert not scaler.copy().is_fitted


def test_standard_scaler_exposes_learned_parameters() -> None:
    scaler = atlas.StandardScaler()

    scaler.fit(FEATURES)

    assert scaler.is_fitted
    assert scaler.mean_.shape == (1,)
    assert scaler.scale_.shape == (1,)
