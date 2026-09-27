//! ML Python binding modules.

mod decision_stump;
mod decision_tree;
mod gaussian_naive_bayes;
mod k_fold;
mod knn;
mod knn_classifier;
mod knn_regressor;
mod linear_regression;
mod logistic_regression;
mod metrics;
mod model;
mod nearest_centroid;
mod perceptron;
mod ridge_regression;
mod scalers;
#[path = "train_test_split.rs"]
mod train_test_split_ops;

use pyo3::{prelude::*, wrap_pyfunction};

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<linear_regression::LinearRegression>()?;
    module.add_class::<ridge_regression::RidgeRegression>()?;
    module.add_class::<logistic_regression::BinaryLogisticRegression>()?;
    module.add_class::<perceptron::BinaryPerceptron>()?;
    module.add_class::<nearest_centroid::NearestCentroidClassifier>()?;
    module.add_class::<gaussian_naive_bayes::GaussianNaiveBayes>()?;
    module.add_class::<knn_classifier::KnnClassifier>()?;
    module.add_class::<knn_regressor::KnnRegressor>()?;
    module.add_class::<decision_stump::DecisionStumpClassifier>()?;
    module.add_class::<decision_tree::BinaryGiniSplit>()?;
    module.add_class::<scalers::StandardScaler>()?;
    module.add_class::<scalers::MinMaxScaler>()?;
    module.add_class::<metrics::ConfusionMatrix>()?;
    module.add_class::<metrics::ClassificationReport>()?;
    module.add_class::<train_test_split_ops::TrainTestSplit>()?;
    module.add_class::<k_fold::KFold>()?;
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
) -> PyResult<Py<decision_tree::BinaryGiniSplit>> {
    decision_tree::evaluate_binary_gini_split(py, feature_values, labels, threshold)
}

#[pyfunction]
fn accuracy(
    py: Python<'_>,
    actual: &Bound<'_, PyAny>,
    predicted: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    metrics::accuracy(py, actual, predicted)
}

#[pyfunction]
fn log_loss(
    py: Python<'_>,
    actual: &Bound<'_, PyAny>,
    probabilities: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    metrics::log_loss(py, actual, probabilities)
}

#[pyfunction]
fn mean_absolute_error(
    py: Python<'_>,
    actual: &Bound<'_, PyAny>,
    predicted: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    metrics::mean_absolute_error(py, actual, predicted)
}

#[pyfunction]
fn mean_squared_error(
    py: Python<'_>,
    actual: &Bound<'_, PyAny>,
    predicted: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    metrics::mean_squared_error(py, actual, predicted)
}

#[pyfunction]
fn r_squared(
    py: Python<'_>,
    actual: &Bound<'_, PyAny>,
    predicted: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    metrics::r_squared(py, actual, predicted)
}

#[pyfunction]
fn confusion_matrix(
    py: Python<'_>,
    actual: &Bound<'_, PyAny>,
    predicted: &Bound<'_, PyAny>,
) -> PyResult<Py<metrics::ConfusionMatrix>> {
    metrics::confusion_matrix(py, actual, predicted)
}

#[pyfunction]
fn classification_report(
    py: Python<'_>,
    actual: &Bound<'_, PyAny>,
    predicted: &Bound<'_, PyAny>,
) -> PyResult<Py<metrics::ClassificationReport>> {
    metrics::classification_report(py, actual, predicted)
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
) -> PyResult<Vec<Py<k_fold::KFold>>> {
    k_fold::k_fold_split(py, features, fold_count, seed)
}

#[pyfunction(signature = (features, labels, fold_count, seed = 0))]
fn stratified_k_fold_split(
    py: Python<'_>,
    features: &Bound<'_, PyAny>,
    labels: &Bound<'_, PyAny>,
    fold_count: usize,
    seed: u64,
) -> PyResult<Vec<Py<k_fold::KFold>>> {
    k_fold::stratified_k_fold_split(py, features, labels, fold_count, seed)
}
