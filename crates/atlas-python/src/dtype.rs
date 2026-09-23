use pyo3::{exceptions::PyValueError, prelude::*, types::PyModule};

macro_rules! with_dtype {
    ($dtype:expr, all |$ty:ident| $body:expr) => {{
        match $dtype {
            DType::Bool => {
                type $ty = bool;
                $body
            }
            DType::Int8 => {
                type $ty = i8;
                $body
            }
            DType::Int16 => {
                type $ty = i16;
                $body
            }
            DType::Int32 => {
                type $ty = i32;
                $body
            }
            DType::Int64 => {
                type $ty = i64;
                $body
            }
            DType::UInt8 => {
                type $ty = u8;
                $body
            }
            DType::UInt16 => {
                type $ty = u16;
                $body
            }
            DType::UInt32 => {
                type $ty = u32;
                $body
            }
            DType::UInt64 => {
                type $ty = u64;
                $body
            }
            DType::Float32 => {
                type $ty = f32;
                $body
            }
            DType::Float64 => {
                type $ty = f64;
                $body
            }
        }
    }};
    ($dtype:expr, numeric |$ty:ident| $body:expr) => {{
        match $dtype {
            DType::Bool => unreachable!("numeric dtype dispatch excludes bool"),
            DType::Int8 => {
                type $ty = i8;
                $body
            }
            DType::Int16 => {
                type $ty = i16;
                $body
            }
            DType::Int32 => {
                type $ty = i32;
                $body
            }
            DType::Int64 => {
                type $ty = i64;
                $body
            }
            DType::UInt8 => {
                type $ty = u8;
                $body
            }
            DType::UInt16 => {
                type $ty = u16;
                $body
            }
            DType::UInt32 => {
                type $ty = u32;
                $body
            }
            DType::UInt64 => {
                type $ty = u64;
                $body
            }
            DType::Float32 => {
                type $ty = f32;
                $body
            }
            DType::Float64 => {
                type $ty = f64;
                $body
            }
        }
    }};
}

pub(crate) use with_dtype;

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
    pub(crate) fn parse(py: Python<'_>, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let name = match dtype {
            Some(dtype) => PyModule::import(py, "numpy")?
                .getattr("dtype")?
                .call1((dtype,))?
                .getattr("name")?
                .extract::<String>()?,
            None => "float64".to_owned(),
        };

        match name.as_str() {
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
            _ => Err(PyValueError::new_err(format!(
                "unsupported dtype {name:?}; expected bool, int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, or float64"
            ))),
        }
    }
}
