use std::fmt;

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

impl DType {
    pub fn of<T: RuntimeDType>() -> Self {
        T::DTYPE
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
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

pub trait RuntimeDType: 'static {
    const DTYPE: DType;
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

#[cfg(test)]
mod tests {
    use super::{DType, RuntimeDType};

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
}
