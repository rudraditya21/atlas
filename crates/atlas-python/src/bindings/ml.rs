//! ML Python bindings.

use pyo3::{prelude::*, wrap_pyfunction};

use crate::{
    decision_stump_ops, decision_tree_ops, gaussian_naive_bayes_ops, k_fold_ops,
    knn_classifier_ops, knn_regressor_ops, linear_regression_ops, logistic_regression_ops,
    metrics_ops, nearest_centroid_ops, perceptron_ops, ridge_regression_ops, scaler_ops,
    train_test_split_ops,
};

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<linear_regression_ops::LinearRegression>()?;
    module.add_class::<ridge_regression_ops::RidgeRegression>()?;
    module.add_class::<logistic_regression_ops::BinaryLogisticRegression>()?;
    module.add_class::<perceptron_ops::BinaryPerceptron>()?;
    module.add_class::<nearest_centroid_ops::NearestCentroidClassifier>()?;
    module.add_class::<gaussian_naive_bayes_ops::GaussianNaiveBayes>()?;
    module.add_class::<knn_classifier_ops::KnnClassifier>()?;
    module.add_class::<knn_regressor_ops::KnnRegressor>()?;
    module.add_class::<decision_stump_ops::DecisionStumpClassifier>()?;
    module.add_class::<decision_tree_ops::BinaryGiniSplit>()?;
    module.add_class::<scaler_ops::StandardScaler>()?;
    module.add_class::<scaler_ops::MinMaxScaler>()?;
    module.add_class::<metrics_ops::ConfusionMatrix>()?;
    module.add_class::<metrics_ops::ClassificationReport>()?;
    module.add_class::<train_test_split_ops::TrainTestSplit>()?;
    module.add_class::<k_fold_ops::KFold>()?;
    module.add_function(wrap_pyfunction!(evaluate_binary_gini_split, module)?)?;
    module.add_function(wrap_pyfunction!(accuracy, module)?)?;
    module.add_function(wrap_pyfunction!(log_loss, module)?)?;
    module.add_function(wrap_pyfunction!(mean_absolute_error, module)?)?;
    module.add_function(wrap_pyfunction!(mean_squared_error, module)?)?;
    module.add_function(wrap_pyfunction!(r_squared, module)?)?;
    module.add_function(wrap_pyfunction!(confusion_matrix, module)?)?;
    module.add_function(wrap_pyfunction!(classification_report, module)?)?;
    module.add_function(wrap_pyfunction!(train_test_split, module)?)?;
    module.add_function(wrap_pyfunction!(k_fold_split, module)?)?;
    module.add_function(wrap_pyfunction!(stratified_k_fold_split, module)?)?;
    Ok(())
}

#[pyfunction]
fn evaluate_binary_gini_split(
    py: Python<'_>,
    feature_values: &Bound<'_, PyAny>,
    labels: &Bound<'_, PyAny>,
    threshold: f64,
) -> PyResult<Py<decision_tree_ops::BinaryGiniSplit>> {
    decision_tree_ops::evaluate_binary_gini_split(py, feature_values, labels, threshold)
}

#[pyfunction]
fn accuracy(
    py: Python<'_>,
    actual: &Bound<'_, PyAny>,
    predicted: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    metrics_ops::accuracy(py, actual, predicted)
}

#[pyfunction]
fn log_loss(
    py: Python<'_>,
    actual: &Bound<'_, PyAny>,
    probabilities: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    metrics_ops::log_loss(py, actual, probabilities)
}

#[pyfunction]
fn mean_absolute_error(
    py: Python<'_>,
    actual: &Bound<'_, PyAny>,
    predicted: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    metrics_ops::mean_absolute_error(py, actual, predicted)
}

#[pyfunction]
fn mean_squared_error(
    py: Python<'_>,
    actual: &Bound<'_, PyAny>,
    predicted: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    metrics_ops::mean_squared_error(py, actual, predicted)
}

#[pyfunction]
fn r_squared(
    py: Python<'_>,
    actual: &Bound<'_, PyAny>,
    predicted: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    metrics_ops::r_squared(py, actual, predicted)
}

#[pyfunction]
fn confusion_matrix(
    py: Python<'_>,
    actual: &Bound<'_, PyAny>,
    predicted: &Bound<'_, PyAny>,
) -> PyResult<Py<metrics_ops::ConfusionMatrix>> {
    metrics_ops::confusion_matrix(py, actual, predicted)
}

#[pyfunction]
fn classification_report(
    py: Python<'_>,
    actual: &Bound<'_, PyAny>,
    predicted: &Bound<'_, PyAny>,
) -> PyResult<Py<metrics_ops::ClassificationReport>> {
    metrics_ops::classification_report(py, actual, predicted)
}

#[pyfunction(signature = (features, targets, test_ratio = 0.25, seed = 0))]
fn train_test_split(
    py: Python<'_>,
    features: &Bound<'_, PyAny>,
    targets: &Bound<'_, PyAny>,
    test_ratio: f64,
    seed: u64,
) -> PyResult<Py<train_test_split_ops::TrainTestSplit>> {
    train_test_split_ops::train_test_split(py, features, targets, test_ratio, seed)
}

#[pyfunction(signature = (features, fold_count, seed = 0))]
fn k_fold_split(
    py: Python<'_>,
    features: &Bound<'_, PyAny>,
    fold_count: usize,
    seed: u64,
) -> PyResult<Vec<Py<k_fold_ops::KFold>>> {
    k_fold_ops::k_fold_split(py, features, fold_count, seed)
}

#[pyfunction(signature = (features, labels, fold_count, seed = 0))]
fn stratified_k_fold_split(
    py: Python<'_>,
    features: &Bound<'_, PyAny>,
    labels: &Bound<'_, PyAny>,
    fold_count: usize,
    seed: u64,
) -> PyResult<Vec<Py<k_fold_ops::KFold>>> {
    k_fold_ops::stratified_k_fold_split(py, features, labels, fold_count, seed)
}
