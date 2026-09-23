use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::{array, gil};

pub(crate) fn roll(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    shift: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let (shifts, axes) = roll_spec(py, shift, axis)?;
    array::require_numpy_array(py, value)?;
    let dtype: String = value.getattr("dtype")?.getattr("name")?.extract()?;

    macro_rules! apply {
        ($ty:ty) => {{
            let array = array::from_numpy(array::readonly_from_python::<$ty>(py, value)?)?;
            let result = gil::without_gil(py, move || match axes {
                Some(axes) => roll_axes(array, &shifts, &axes),
                None => {
                    let shape = array.shape().to_vec();
                    array
                        .flatten()
                        .roll(shifts[0], 0)
                        .and_then(|rolled| rolled.reshape(shape).map(|view| view.to_owned()))
                }
            })
            .map_err(|error| crate::error::ndarray(py, error))?;
            Ok(array::to_numpy_owned(py, result)?.into_any().unbind())
        }};
    }

    match dtype.as_str() {
        "bool" => apply!(bool),
        "int8" => apply!(i8),
        "int16" => apply!(i16),
        "int32" => apply!(i32),
        "int64" => apply!(i64),
        "uint8" => apply!(u8),
        "uint16" => apply!(u16),
        "uint32" => apply!(u32),
        "uint64" => apply!(u64),
        "float32" => apply!(f32),
        "float64" => apply!(f64),
        _ => Err(PyTypeError::new_err(format!("unsupported NumPy dtype {dtype}"))),
    }
}

fn roll_spec(
    py: Python<'_>,
    shift: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
) -> PyResult<(Vec<i64>, Option<Vec<i64>>)> {
    let mut shifts = integer_values(shift)?;
    let Some(axis) = axis else {
        if shifts.len() != 1 {
            return Err(roll_error(py, "shift must be scalar when axis is None"));
        }
        return Ok((shifts, None));
    };
    let mut axes = integer_values(axis)?;

    if shifts.len() == 1 {
        shifts = vec![shifts[0]; axes.len()];
    } else if axes.len() == 1 {
        axes = vec![axes[0]; shifts.len()];
    } else if shifts.len() != axes.len() {
        return Err(roll_error(py, "shift and axis must have matching lengths or be scalar"));
    }

    Ok((shifts, Some(axes)))
}

fn integer_values(value: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    value.extract::<i64>().map(|value| vec![value]).or_else(|_| value.extract())
}

fn roll_axes<T>(
    mut array: atlas_ndarray::NDArray<T>,
    shifts: &[i64],
    axes: &[i64],
) -> atlas_ndarray::AtlasNdResult<atlas_ndarray::NDArray<T>>
where
    T: atlas_ndarray::ArrayElement,
{
    for (&shift, &axis) in shifts.iter().zip(axes) {
        array = array.roll(shift, axis)?;
    }
    Ok(array)
}

fn roll_error(py: Python<'_>, reason: &'static str) -> PyErr {
    crate::error::ndarray(py, atlas_ndarray::AtlasNdError::InvalidArgument { op: "roll", reason })
}
