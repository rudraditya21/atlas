//! Random Python bindings.

use pyo3::prelude::*;

use crate::generator;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<generator::Generator>()
}
