use atlas_ndarray::ScalarValue;
use numpy::{PyArrayDescr, PyArrayDescrMethods, dtype};
use pyo3::{
    exceptions::{PyOverflowError, PyTypeError},
    prelude::*,
    types::{PyBool, PyFloat, PyInt},
};

pub(crate) fn from_python(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<ScalarValue> {
    if value.is_instance_of::<PyBool>() {
        return Ok(ScalarValue::Bool(value.extract()?));
    }

    if value.is_instance_of::<PyInt>() {
        return integer_scalar(value);
    }

    if value.is_instance_of::<PyFloat>() {
        return Ok(ScalarValue::F64(value.extract()?));
    }

    if let Some(value) = numpy_scalar(py, value)? {
        return Ok(value);
    }

    Err(PyTypeError::new_err("expected a bool, integer, float, or supported NumPy scalar"))
}

fn numpy_scalar(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Option<ScalarValue>> {
    let numpy = PyModule::import(py, "numpy")?;
    if !value.is_instance(&numpy.getattr("generic")?)? {
        return Ok(None);
    }

    let scalar_dtype = value.getattr("dtype")?.cast_into::<PyArrayDescr>()?;
    let scalar = if scalar_dtype.is_equiv_to(&dtype::<bool>(py)) {
        ScalarValue::Bool(value.extract()?)
    } else if scalar_dtype.is_equiv_to(&dtype::<i8>(py)) {
        ScalarValue::I8(value.extract()?)
    } else if scalar_dtype.is_equiv_to(&dtype::<i16>(py)) {
        ScalarValue::I16(value.extract()?)
    } else if scalar_dtype.is_equiv_to(&dtype::<i32>(py)) {
        ScalarValue::I32(value.extract()?)
    } else if scalar_dtype.is_equiv_to(&dtype::<i64>(py)) {
        ScalarValue::I64(value.extract()?)
    } else if scalar_dtype.is_equiv_to(&dtype::<u8>(py)) {
        ScalarValue::U8(value.extract()?)
    } else if scalar_dtype.is_equiv_to(&dtype::<u16>(py)) {
        ScalarValue::U16(value.extract()?)
    } else if scalar_dtype.is_equiv_to(&dtype::<u32>(py)) {
        ScalarValue::U32(value.extract()?)
    } else if scalar_dtype.is_equiv_to(&dtype::<u64>(py)) {
        ScalarValue::U64(value.extract()?)
    } else if scalar_dtype.is_equiv_to(&dtype::<f32>(py)) {
        ScalarValue::F32(value.extract()?)
    } else if scalar_dtype.is_equiv_to(&dtype::<f64>(py)) {
        ScalarValue::F64(value.extract()?)
    } else {
        return Err(PyTypeError::new_err("unsupported NumPy scalar dtype"));
    };

    Ok(Some(scalar))
}

fn integer_scalar(value: &Bound<'_, PyAny>) -> PyResult<ScalarValue> {
    match value.extract::<i64>() {
        Ok(value) => Ok(ScalarValue::I64(value)),
        Err(_) => value
            .extract::<u64>()
            .map(ScalarValue::U64)
            .map_err(|_| PyOverflowError::new_err("integer is outside Atlas's supported range")),
    }
}

pub(crate) fn kind(value: ScalarValue) -> &'static str {
    match value {
        ScalarValue::Bool(_) => "bool",
        ScalarValue::I8(_) => "int8",
        ScalarValue::I16(_) => "int16",
        ScalarValue::I32(_) => "int32",
        ScalarValue::I64(_) => "int64",
        ScalarValue::Isize(_) => "isize",
        ScalarValue::U8(_) => "uint8",
        ScalarValue::U16(_) => "uint16",
        ScalarValue::U32(_) => "uint32",
        ScalarValue::U64(_) => "uint64",
        ScalarValue::Usize(_) => "usize",
        ScalarValue::F32(_) => "float32",
        ScalarValue::F64(_) => "float64",
    }
}
