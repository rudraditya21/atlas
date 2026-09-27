//! Shared Python binding result types.

use pyo3::prelude::*;

pub(crate) type PyObjectResult = PyResult<Py<PyAny>>;
