//! Native Python bindings for Atlas.
//!
//! This crate owns the Python extension boundary. Numerical implementations remain in the
//! Atlas library crates and will be registered here incrementally.

use pyo3::prelude::*;

#[path = "bindings/arrays.rs"]
mod array_bindings;
#[path = "bindings/support.rs"]
mod support_bindings;

#[path = "indexing/argpartition.rs"]
mod argpartition_ops;
#[path = "indexing/argsort.rs"]
mod argsort_ops;
#[path = "support/array.rs"]
mod array;
#[path = "manipulation/concat.rs"]
mod concat_ops;
mod constructors;
#[path = "support/error.rs"]
mod error;
#[path = "manipulation/flatten.rs"]
mod flatten_ops;
#[path = "manipulation/flip.rs"]
mod flip_ops;
#[path = "support/gil.rs"]
mod gil;
mod interop;
mod linalg;
mod ml;
mod operations;
#[path = "manipulation/pad.rs"]
mod pad_ops;
#[path = "indexing/partition.rs"]
mod partition_ops;
#[path = "indexing/put.rs"]
mod put_ops;
#[path = "support/dtype.rs"]
mod python_dtype;
mod random;
#[path = "manipulation/ravel.rs"]
mod ravel_ops;
mod reductions;
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
#[path = "indexing/sort.rs"]
mod sort_ops;
#[path = "manipulation/split.rs"]
mod split_ops;
#[path = "manipulation/stack.rs"]
mod stack_ops;
mod statistics;
#[path = "indexing/take.rs"]
mod take_ops;
#[cfg(feature = "test-support")]
#[path = "support/test_support.rs"]
mod test_support;
#[path = "manipulation/tile.rs"]
mod tile_ops;
#[path = "indexing/unique.rs"]
mod unique_ops;

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    support_bindings::register(module)?;
    random::register(module)?;
    constructors::register(module)?;
    operations::register(module)?;
    reductions::register(module)?;
    array_bindings::register(module)?;
    linalg::register(module)?;
    statistics::register(module)?;
    ml::register(module)?;
    interop::register(module)?;
    Ok(())
}
