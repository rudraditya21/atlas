//! Transitional array binding registrar.

use pyo3::prelude::*;

pub(crate) fn register(_: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
