//! Native Python bindings for Atlas.
//!
//! This crate owns the Python extension boundary. Numerical implementations remain in the
//! Atlas library crates and will be registered here incrementally.

use pyo3::prelude::*;

#[path = "bindings/arrays.rs"]
mod array_bindings;
#[path = "bindings/interop.rs"]
mod interop_bindings;
#[path = "bindings/linalg.rs"]
mod linalg_bindings;
#[path = "bindings/ml.rs"]
mod ml_bindings;
#[path = "bindings/random.rs"]
mod random_bindings;
#[path = "bindings/statistics.rs"]
mod statistics_bindings;
#[path = "bindings/support.rs"]
mod support_bindings;

#[path = "indexing/argpartition.rs"]
mod argpartition_ops;
#[path = "indexing/argsort.rs"]
mod argsort_ops;
#[path = "operations/arithmetic.rs"]
mod arithmetic;
#[path = "support/array.rs"]
mod array;
#[path = "interop/arrow.rs"]
mod arrow_ops;
#[path = "operations/bitwise.rs"]
mod bitwise;
#[path = "constructors/casting.rs"]
mod casting;
#[path = "linalg/cholesky.rs"]
mod cholesky_ops;
#[path = "operations/clip.rs"]
mod clip_ops;
#[path = "operations/close.rs"]
mod close;
#[path = "manipulation/concat.rs"]
mod concat_ops;
#[path = "linalg/conjugate_gradient.rs"]
mod conjugate_gradient_ops;
#[path = "constructors/constructors.rs"]
mod constructors;
#[path = "ml/decision_stump.rs"]
mod decision_stump_ops;
#[path = "ml/decision_tree.rs"]
mod decision_tree_ops;
#[path = "linalg/determinant.rs"]
mod determinant_ops;
#[path = "linalg/diag.rs"]
mod diag_ops;
#[path = "linalg/dot.rs"]
mod dot_ops;
#[path = "linalg/eigen.rs"]
mod eigen_ops;
#[path = "support/error.rs"]
mod error;
#[path = "manipulation/flatten.rs"]
mod flatten_ops;
#[path = "manipulation/flip.rs"]
mod flip_ops;
#[path = "ml/gaussian_naive_bayes.rs"]
mod gaussian_naive_bayes_ops;
#[path = "random/generator.rs"]
mod generator;
#[path = "support/gil.rs"]
mod gil;
#[path = "linalg/inverse.rs"]
mod inverse_ops;
#[path = "ml/k_fold.rs"]
mod k_fold_ops;
#[path = "ml/knn_classifier.rs"]
mod knn_classifier_ops;
#[path = "ml/knn_regressor.rs"]
mod knn_regressor_ops;
#[path = "ml/knn.rs"]
mod knn_support;
#[path = "statistics/kurtosis.rs"]
mod kurtosis_ops;
#[path = "ml/linear_regression.rs"]
mod linear_regression_ops;
#[path = "operations/logical.rs"]
mod logical;
#[path = "ml/logistic_regression.rs"]
mod logistic_regression_ops;
#[path = "linalg/matmul.rs"]
mod matmul_ops;
#[path = "linalg/matrix_norm.rs"]
mod matrix_norm_ops;
#[path = "support/metadata.rs"]
mod metadata;
#[path = "ml/metrics.rs"]
mod metrics_ops;
#[allow(dead_code, reason = "model classes are registered incrementally")]
#[path = "ml/model.rs"]
mod model_support;
#[path = "ml/nearest_centroid.rs"]
mod nearest_centroid_ops;
#[path = "linalg/norm.rs"]
mod norm_ops;
#[path = "manipulation/pad.rs"]
mod pad_ops;
#[path = "statistics/pairwise.rs"]
mod pairwise_ops;
#[path = "indexing/partition.rs"]
mod partition_ops;
#[path = "ml/perceptron.rs"]
mod perceptron_ops;
#[path = "indexing/put.rs"]
mod put_ops;
#[path = "support/dtype.rs"]
mod python_dtype;
#[path = "linalg/qr.rs"]
mod qr_ops;
#[path = "statistics/quantile.rs"]
mod quantile_ops;
#[path = "manipulation/ravel.rs"]
mod ravel_ops;
#[path = "reductions/reduction.rs"]
mod reduction;
#[path = "manipulation/repeat.rs"]
mod repeat_ops;
#[path = "ml/ridge_regression.rs"]
mod ridge_regression_ops;
#[path = "manipulation/roll.rs"]
mod roll_ops;
#[cfg(feature = "test-support")]
#[path = "support/scalar.rs"]
mod scalar;
#[path = "ml/scalers.rs"]
mod scaler_ops;
#[path = "indexing/searchsorted.rs"]
mod searchsorted_ops;
#[path = "manipulation/shape_ops.rs"]
mod shape_ops;
#[path = "statistics/skewness.rs"]
mod skewness_ops;
#[path = "linalg/slogdet.rs"]
mod slogdet_ops;
#[path = "linalg/solve.rs"]
mod solve_ops;
#[path = "indexing/sort.rs"]
mod sort_ops;
#[path = "manipulation/split.rs"]
mod split_ops;
#[path = "manipulation/stack.rs"]
mod stack_ops;
#[path = "indexing/take.rs"]
mod take_ops;
#[cfg(feature = "test-support")]
#[path = "support/test_support.rs"]
mod test_support;
#[path = "manipulation/tile.rs"]
mod tile_ops;
#[path = "linalg/trace.rs"]
mod trace_ops;
#[path = "ml/train_test_split.rs"]
mod train_test_split_ops;
#[path = "operations/unary.rs"]
mod unary;
#[path = "indexing/unique.rs"]
mod unique_ops;
#[path = "statistics/weighted.rs"]
mod weighted_ops;
#[path = "operations/where_ops.rs"]
mod where_ops;

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    support_bindings::register(module)?;
    random_bindings::register(module)?;
    array_bindings::register(module)?;
    linalg_bindings::register(module)?;
    statistics_bindings::register(module)?;
    ml_bindings::register(module)?;
    interop_bindings::register(module)?;
    Ok(())
}
