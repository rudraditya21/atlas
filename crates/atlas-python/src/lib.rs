//! Native Python bindings for Atlas.
//!
//! This crate owns the Python extension boundary. Numerical implementations remain in the
//! Atlas library crates and will be registered here incrementally.

use pyo3::prelude::*;

#[path = "bindings/arrays.rs"]
mod array_bindings;
#[path = "bindings/linalg.rs"]
mod linalg_bindings;
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
#[path = "random/generator.rs"]
mod generator;
#[path = "support/gil.rs"]
mod gil;
mod interop;
#[path = "linalg/inverse.rs"]
mod inverse_ops;
#[path = "statistics/kurtosis.rs"]
mod kurtosis_ops;
#[path = "operations/logical.rs"]
mod logical;
#[path = "linalg/matmul.rs"]
mod matmul_ops;
#[path = "linalg/matrix_norm.rs"]
mod matrix_norm_ops;
#[path = "support/metadata.rs"]
mod metadata;
mod ml;
#[path = "linalg/norm.rs"]
mod norm_ops;
#[path = "manipulation/pad.rs"]
mod pad_ops;
#[path = "statistics/pairwise.rs"]
mod pairwise_ops;
#[path = "indexing/partition.rs"]
mod partition_ops;
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
#[path = "support/results.rs"]
mod results;
#[path = "manipulation/roll.rs"]
mod roll_ops;
#[cfg(feature = "test-support")]
#[path = "support/scalar.rs"]
mod scalar;
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
    ml::register(module)?;
    interop::register(module)?;
    Ok(())
}
