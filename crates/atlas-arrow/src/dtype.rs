use atlas_ndarray::{ArrayElement, DType};

mod sealed {
    pub trait Sealed {}

    macro_rules! impl_sealed {
        ($($ty:ty),+ $(,)?) => { $(impl Sealed for $ty {})+ };
    }

    impl_sealed!(bool, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64);
}

/// Primitive Atlas dtypes currently supported by the Arrow interchange boundary.
///
/// Strings, binary data, temporal values, nested data, decimal values, and null-bearing arrays
/// are intentionally out of scope for the initial interchange surface.
pub trait InterchangeDType: ArrayElement + sealed::Sealed {
    /// Atlas dtype represented by this primitive interchange value.
    const DTYPE: DType;
}

macro_rules! impl_interchange_dtype {
    ($($ty:ty => $dtype:expr),+ $(,)?) => {
        $(
            impl InterchangeDType for $ty {
                const DTYPE: DType = $dtype;
            }
        )+
    };
}

impl_interchange_dtype!(
    bool => DType::Bool,
    i8 => DType::I8,
    i16 => DType::I16,
    i32 => DType::I32,
    i64 => DType::I64,
    isize => DType::Isize,
    u8 => DType::U8,
    u16 => DType::U16,
    u32 => DType::U32,
    u64 => DType::U64,
    usize => DType::Usize,
    f32 => DType::F32,
    f64 => DType::F64,
);

#[cfg(test)]
mod tests {
    use atlas_ndarray::DType;

    use super::InterchangeDType;

    #[test]
    fn primitive_interchange_dtypes_map_to_atlas_dtypes() {
        assert_eq!(<bool as InterchangeDType>::DTYPE, DType::Bool);
        assert_eq!(<i64 as InterchangeDType>::DTYPE, DType::I64);
        assert_eq!(<u64 as InterchangeDType>::DTYPE, DType::U64);
        assert_eq!(<f64 as InterchangeDType>::DTYPE, DType::F64);
    }
}
