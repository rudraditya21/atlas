use std::{fmt, mem::size_of};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DTypeKind {
    Bool,
    SignedInteger,
    UnsignedInteger,
    Float,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    I8,
    I16,
    I32,
    I64,
    Isize,
    U8,
    U16,
    U32,
    U64,
    Usize,
    F32,
    F64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalarValue {
    Bool(bool),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    Isize(isize),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    Usize(usize),
    F32(f32),
    F64(f64),
}

impl DType {
    pub fn of<T: RuntimeDType>() -> Self {
        T::DTYPE
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "bool" => Some(Self::Bool),
            "int8" => Some(Self::I8),
            "int16" => Some(Self::I16),
            "int32" => Some(Self::I32),
            "int64" => Some(Self::I64),
            "intp" => Some(Self::Isize),
            "uint8" => Some(Self::U8),
            "uint16" => Some(Self::U16),
            "uint32" => Some(Self::U32),
            "uint64" => Some(Self::U64),
            "uintp" => Some(Self::Usize),
            "float32" => Some(Self::F32),
            "float64" => Some(Self::F64),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::I8 => "int8",
            Self::I16 => "int16",
            Self::I32 => "int32",
            Self::I64 => "int64",
            Self::Isize => "intp",
            Self::U8 => "uint8",
            Self::U16 => "uint16",
            Self::U32 => "uint32",
            Self::U64 => "uint64",
            Self::Usize => "uintp",
            Self::F32 => "float32",
            Self::F64 => "float64",
        }
    }

    pub fn kind(self) -> DTypeKind {
        match self {
            Self::Bool => DTypeKind::Bool,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::Isize => DTypeKind::SignedInteger,
            Self::U8 | Self::U16 | Self::U32 | Self::U64 | Self::Usize => {
                DTypeKind::UnsignedInteger
            }
            Self::F32 | Self::F64 => DTypeKind::Float,
        }
    }

    pub fn itemsize(self) -> usize {
        match self {
            Self::Bool => size_of::<bool>(),
            Self::I8 => size_of::<i8>(),
            Self::I16 => size_of::<i16>(),
            Self::I32 => size_of::<i32>(),
            Self::I64 => size_of::<i64>(),
            Self::Isize => size_of::<isize>(),
            Self::U8 => size_of::<u8>(),
            Self::U16 => size_of::<u16>(),
            Self::U32 => size_of::<u32>(),
            Self::U64 => size_of::<u64>(),
            Self::Usize => size_of::<usize>(),
            Self::F32 => size_of::<f32>(),
            Self::F64 => size_of::<f64>(),
        }
    }

    pub fn is_bool(self) -> bool {
        matches!(self, Self::Bool)
    }

    pub fn is_integer(self) -> bool {
        matches!(self.kind(), DTypeKind::SignedInteger | DTypeKind::UnsignedInteger)
    }

    pub fn is_signed_integer(self) -> bool {
        matches!(self.kind(), DTypeKind::SignedInteger)
    }

    pub fn is_unsigned_integer(self) -> bool {
        matches!(self.kind(), DTypeKind::UnsignedInteger)
    }

    pub fn is_float(self) -> bool {
        matches!(self.kind(), DTypeKind::Float)
    }

    pub fn is_signed(self) -> bool {
        self.is_signed_integer() || self.is_float()
    }

    pub fn matches<T: RuntimeDType>(self) -> bool {
        self == T::DTYPE
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl ScalarValue {
    pub fn dtype(self) -> DType {
        match self {
            Self::Bool(_) => DType::Bool,
            Self::I8(_) => DType::I8,
            Self::I16(_) => DType::I16,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::Isize(_) => DType::Isize,
            Self::U8(_) => DType::U8,
            Self::U16(_) => DType::U16,
            Self::U32(_) => DType::U32,
            Self::U64(_) => DType::U64,
            Self::Usize(_) => DType::Usize,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
        }
    }
}

pub trait RuntimeDType: 'static {
    const DTYPE: DType;

    fn dtype() -> DType
    where
        Self: Sized,
    {
        Self::DTYPE
    }

    fn itemsize() -> usize
    where
        Self: Sized,
    {
        size_of::<Self>()
    }
}

pub trait RuntimeScalar: RuntimeDType + Copy {
    fn infer_dtype(self) -> DType {
        Self::DTYPE
    }

    fn into_scalar_value(self) -> ScalarValue;
}

pub fn infer_scalar_dtype<T: RuntimeScalar>(value: T) -> DType {
    value.infer_dtype()
}

macro_rules! impl_runtime_dtype {
    ($dtype:ident => $($ty:ty),+ $(,)?) => {
        $(
            impl RuntimeDType for $ty {
                const DTYPE: DType = DType::$dtype;
            }
        )+
    };
}

impl_runtime_dtype!(Bool => bool);
impl_runtime_dtype!(I8 => i8);
impl_runtime_dtype!(I16 => i16);
impl_runtime_dtype!(I32 => i32);
impl_runtime_dtype!(I64 => i64);
impl_runtime_dtype!(Isize => isize);
impl_runtime_dtype!(U8 => u8);
impl_runtime_dtype!(U16 => u16);
impl_runtime_dtype!(U32 => u32);
impl_runtime_dtype!(U64 => u64);
impl_runtime_dtype!(Usize => usize);
impl_runtime_dtype!(F32 => f32);
impl_runtime_dtype!(F64 => f64);

macro_rules! impl_runtime_scalar {
    ($variant:ident => $($ty:ty),+ $(,)?) => {
        $(
            impl RuntimeScalar for $ty {
                fn into_scalar_value(self) -> ScalarValue {
                    ScalarValue::$variant(self)
                }
            }

            impl From<$ty> for ScalarValue {
                fn from(value: $ty) -> Self {
                    ScalarValue::$variant(value)
                }
            }
        )+
    };
}

impl_runtime_scalar!(Bool => bool);
impl_runtime_scalar!(I8 => i8);
impl_runtime_scalar!(I16 => i16);
impl_runtime_scalar!(I32 => i32);
impl_runtime_scalar!(I64 => i64);
impl_runtime_scalar!(Isize => isize);
impl_runtime_scalar!(U8 => u8);
impl_runtime_scalar!(U16 => u16);
impl_runtime_scalar!(U32 => u32);
impl_runtime_scalar!(U64 => u64);
impl_runtime_scalar!(Usize => usize);
impl_runtime_scalar!(F32 => f32);
impl_runtime_scalar!(F64 => f64);

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use super::{DType, DTypeKind, RuntimeDType, RuntimeScalar, ScalarValue, infer_scalar_dtype};

    #[test]
    fn dtype_of_maps_primitive_scalars_to_runtime_variants() {
        assert_eq!(DType::of::<bool>(), DType::Bool);
        assert_eq!(DType::of::<i32>(), DType::I32);
        assert_eq!(DType::of::<u64>(), DType::U64);
        assert_eq!(DType::of::<f32>(), DType::F32);
        assert_eq!(<f64 as RuntimeDType>::DTYPE, DType::F64);
    }

    #[test]
    fn dtype_display_uses_numpy_style_names() {
        assert_eq!(DType::Bool.to_string(), "bool");
        assert_eq!(DType::I64.to_string(), "int64");
        assert_eq!(DType::Usize.to_string(), "uintp");
        assert_eq!(DType::F32.to_string(), "float32");
    }

    #[test]
    fn dtype_from_name_round_trips_supported_values() {
        assert_eq!(DType::from_name("bool"), Some(DType::Bool));
        assert_eq!(DType::from_name("int32"), Some(DType::I32));
        assert_eq!(DType::from_name("uintp"), Some(DType::Usize));
        assert_eq!(DType::from_name("float64"), Some(DType::F64));
        assert_eq!(DType::from_name("complex64"), None);
    }

    #[test]
    fn dtype_metadata_helpers_classify_kinds_and_sizes() {
        assert_eq!(DType::Bool.kind(), DTypeKind::Bool);
        assert_eq!(DType::I16.kind(), DTypeKind::SignedInteger);
        assert_eq!(DType::U32.kind(), DTypeKind::UnsignedInteger);
        assert_eq!(DType::F64.kind(), DTypeKind::Float);

        assert_eq!(DType::Bool.itemsize(), size_of::<bool>());
        assert_eq!(DType::I64.itemsize(), size_of::<i64>());
        assert_eq!(DType::Usize.itemsize(), size_of::<usize>());
        assert_eq!(DType::F32.itemsize(), size_of::<f32>());
    }

    #[test]
    fn dtype_predicates_capture_bool_integer_float_and_signedness() {
        assert!(DType::Bool.is_bool());
        assert!(DType::I32.is_integer());
        assert!(DType::U64.is_integer());
        assert!(DType::I32.is_signed_integer());
        assert!(DType::U64.is_unsigned_integer());
        assert!(DType::F64.is_float());
        assert!(DType::F64.is_signed());
        assert!(!DType::Bool.is_signed());
        assert!(!DType::U16.is_signed());
    }

    #[test]
    fn dtype_matching_and_runtime_helpers_reuse_trait_mapping() {
        assert!(DType::I32.matches::<i32>());
        assert!(!DType::I32.matches::<u32>());
        assert_eq!(<u64 as RuntimeDType>::dtype(), DType::U64);
        assert_eq!(<f32 as RuntimeDType>::itemsize(), size_of::<f32>());
    }

    #[test]
    fn scalar_dtype_inference_follows_runtime_scalar_mapping() {
        assert_eq!(infer_scalar_dtype(true), DType::Bool);
        assert_eq!(infer_scalar_dtype(7_i32), DType::I32);
        assert_eq!(infer_scalar_dtype(9_u64), DType::U64);
        assert_eq!(infer_scalar_dtype(1.5_f32), DType::F32);
    }

    #[test]
    fn scalar_values_preserve_value_and_runtime_dtype() {
        let int_value = ScalarValue::from(42_i64);
        let float_value = 3.25_f64.into_scalar_value();

        assert_eq!(int_value, ScalarValue::I64(42));
        assert_eq!(int_value.dtype(), DType::I64);
        assert_eq!(float_value, ScalarValue::F64(3.25));
        assert_eq!(float_value.dtype(), DType::F64);
    }
}
