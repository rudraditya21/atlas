use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::PyModule,
};

macro_rules! with_dtype {
    ($dtype:expr, all |$ty:ident| $body:expr) => {{
        use $crate::python_dtype::DType;

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
        use $crate::python_dtype::DType;

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
    ($dtype:expr, signed |$ty:ident| $body:expr) => {{
        use $crate::python_dtype::DType;

        match $dtype {
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
            DType::Float32 => {
                type $ty = f32;
                $body
            }
            DType::Float64 => {
                type $ty = f64;
                $body
            }
            _ => unreachable!("signed dtype dispatch requires a signed or floating-point dtype"),
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

        Self::from_name(&name).ok_or_else(|| {
            PyValueError::new_err(format!(
                "unsupported dtype {name:?}; expected bool, int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, or float64"
            ))
        })
    }

    pub(crate) fn from_numpy_name(name: &str) -> PyResult<Self> {
        Self::from_name(name)
            .ok_or_else(|| PyTypeError::new_err(format!("unsupported NumPy dtype {name}")))
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::Int8 => "int8",
            Self::Int16 => "int16",
            Self::Int32 => "int32",
            Self::Int64 => "int64",
            Self::UInt8 => "uint8",
            Self::UInt16 => "uint16",
            Self::UInt32 => "uint32",
            Self::UInt64 => "uint64",
            Self::Float32 => "float32",
            Self::Float64 => "float64",
        }
    }

    pub(crate) fn is_unsigned(self) -> bool {
        matches!(self, Self::UInt8 | Self::UInt16 | Self::UInt32 | Self::UInt64)
    }

    fn from_name(name: &str) -> Option<Self> {
        match name {
            "bool" => Some(Self::Bool),
            "int8" => Some(Self::Int8),
            "int16" => Some(Self::Int16),
            "int32" => Some(Self::Int32),
            "int64" => Some(Self::Int64),
            "uint8" => Some(Self::UInt8),
            "uint16" => Some(Self::UInt16),
            "uint32" => Some(Self::UInt32),
            "uint64" => Some(Self::UInt64),
            "float32" => Some(Self::Float32),
            "float64" => Some(Self::Float64),
            _ => None,
        }
    }
}
