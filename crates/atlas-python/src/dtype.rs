use pyo3::{exceptions::PyValueError, prelude::*};

#[derive(Clone, Copy)]
pub(crate) enum DType {
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
}

impl DType {
    pub(crate) fn parse(dtype: Option<&str>) -> PyResult<Self> {
        match dtype.unwrap_or("float64") {
            "bool" => Ok(Self::Bool),
            "int8" => Ok(Self::Int8),
            "int16" => Ok(Self::Int16),
            "int32" => Ok(Self::Int32),
            "int64" => Ok(Self::Int64),
            "uint8" => Ok(Self::UInt8),
            "uint16" => Ok(Self::UInt16),
            "uint32" => Ok(Self::UInt32),
            "uint64" => Ok(Self::UInt64),
            "float32" => Ok(Self::Float32),
            "float64" => Ok(Self::Float64),
            dtype => Err(PyValueError::new_err(format!(
                "unsupported dtype {dtype:?}; expected bool, int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, or float64"
            ))),
        }
    }
}
