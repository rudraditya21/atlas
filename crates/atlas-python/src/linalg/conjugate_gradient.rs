use num_traits::ToPrimitive;
use pyo3::prelude::*;

use crate::support::{arrays as array, gil};

#[pyclass(module = "atlas._native")]
pub(crate) struct ConjugateGradientResult {
    solution: Py<PyAny>,
    iterations: usize,
    residual_norm: f64,
    converged: bool,
}

#[pymethods]
impl ConjugateGradientResult {
    #[getter]
    fn solution(&self, py: Python<'_>) -> Py<PyAny> {
        self.solution.clone_ref(py)
    }

    #[getter]
    fn iterations(&self) -> usize {
        self.iterations
    }

    #[getter]
    fn residual_norm(&self) -> f64 {
        self.residual_norm
    }

    #[getter]
    fn converged(&self) -> bool {
        self.converged
    }

    #[getter]
    fn dtype(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        array::metadata_dtype(py, &self.solution)
    }

    #[getter]
    fn shape(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        array::metadata_shape(py, &self.solution)
    }

    #[getter]
    fn ndim(&self, py: Python<'_>) -> PyResult<usize> {
        array::metadata_ndim(py, &self.solution)
    }

    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        array::metadata_len(py, &self.solution)
    }
}

pub(crate) fn conjugate_gradient(
    py: Python<'_>,
    matrix: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
    max_iterations: usize,
    tolerance: f64,
) -> PyResult<Py<ConjugateGradientResult>> {
    array::require_numpy_array(py, matrix)?;
    array::require_numpy_array(py, rhs)?;
    let dtype: String = matrix.getattr("dtype")?.getattr("name")?.extract()?;

    macro_rules! apply {
        ($ty:ty) => {{
            let matrix = array::from_numpy(array::readonly_from_python::<$ty>(py, matrix)?)?;
            let rhs = array::from_numpy(array::readonly_from_python::<$ty>(py, rhs)?)?;
            let result = gil::without_gil(py, move || {
                atlas_linalg::conjugate_gradient_with_diagnostics(
                    &matrix,
                    &rhs,
                    max_iterations,
                    num_traits::cast::<f64, $ty>(tolerance)
                        .expect("f64 tolerance must convert to the target float dtype"),
                )
            })
            .map_err(|error| crate::support::errors::linalg(py, error))?;
            output(py, result)
        }};
    }

    match dtype.as_str() {
        "float32" => apply!(f32),
        "float64" => apply!(f64),
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            "conjugate_gradient requires float32 or float64 NumPy arrays",
        )),
    }
}

fn output<T>(
    py: Python<'_>,
    result: atlas_linalg::ConjugateGradientResult<T>,
) -> PyResult<Py<ConjugateGradientResult>>
where
    T: atlas_ndarray::Numeric + numpy::Element + ToPrimitive,
{
    Py::new(
        py,
        ConjugateGradientResult {
            solution: array::to_numpy_owned(py, result.solution().clone())?.into_any().unbind(),
            iterations: result.iterations(),
            residual_norm: result
                .residual_norm()
                .to_f64()
                .expect("floating-point residual norm must convert to f64"),
            converged: result.converged(),
        },
    )
}
