import pytest

import atlas


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (atlas.LinearRegression, "LinearRegression()"),
        (
            lambda: atlas.RidgeRegression(0.5),
            "RidgeRegression(l2_regularization=0.5)",
        ),
        (
            lambda: atlas.BinaryLogisticRegression(
                learning_rate=0.2,
                max_iterations=25,
                convergence_tolerance=0.001,
                l2_regularization=0.5,
            ),
            "BinaryLogisticRegression(learning_rate=0.2, max_iterations=25, convergence_tolerance=0.001, l2_regularization=0.5)",
        ),
        (
            lambda: atlas.BinaryPerceptron(
                learning_rate=0.2, max_iterations=25, shuffle_seed=7
            ),
            "BinaryPerceptron(learning_rate=0.2, max_iterations=25, shuffle_seed=7)",
        ),
        (atlas.NearestCentroidClassifier, "NearestCentroidClassifier()"),
        (
            lambda: atlas.GaussianNaiveBayes(0.001),
            "GaussianNaiveBayes(variance_smoothing=0.001)",
        ),
        (
            lambda: atlas.KnnClassifier(
                3, search_algorithm="kd_tree", weighting="distance", tree_leaf_size=4
            ),
            "KnnClassifier(k=3, search_algorithm='kd_tree', weighting='distance', tree_leaf_size=4)",
        ),
        (
            lambda: atlas.KnnRegressor(
                3, search_algorithm="ball_tree", weighting="uniform", tree_leaf_size=4
            ),
            "KnnRegressor(k=3, search_algorithm='ball_tree', weighting='uniform', tree_leaf_size=4)",
        ),
        (atlas.DecisionStumpClassifier, "DecisionStumpClassifier()"),
        (atlas.StandardScaler, "StandardScaler()"),
        (
            lambda: atlas.MinMaxScaler(-1.0, 2.0),
            "MinMaxScaler(output_minimum=-1, output_maximum=2)",
        ),
    ],
)
def test_model_representations_include_constructor_configuration(
    factory, expected: str
) -> None:
    assert repr(factory()) == expected


def test_perceptron_representation_preserves_a_disabled_shuffle_policy() -> None:
    assert repr(atlas.BinaryPerceptron()) == (
        "BinaryPerceptron(learning_rate=0.1, max_iterations=1000, shuffle_seed=None)"
    )
