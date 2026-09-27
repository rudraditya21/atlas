//! Random Python binding modules.

mod generator;

use pyo3::prelude::*;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<generator::Generator>()
}
