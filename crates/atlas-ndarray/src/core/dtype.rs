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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CastPolicy {
    Exact,
    Checked,
    Lossy,
    Forbidden,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CastMode {
    Checked,
    Lossy,
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

    pub fn cast_policy_to(self, target: DType) -> CastPolicy {
        if self == target {
            return CastPolicy::Exact;
        }

        if self.is_bool() {
            return CastPolicy::Exact;
        }

        if target.is_bool() {
            return CastPolicy::Lossy;
        }

        match (self.kind(), target.kind()) {
            (DTypeKind::SignedInteger, DTypeKind::SignedInteger)
            | (DTypeKind::UnsignedInteger, DTypeKind::UnsignedInteger)
            | (DTypeKind::SignedInteger, DTypeKind::UnsignedInteger)
            | (DTypeKind::UnsignedInteger, DTypeKind::SignedInteger) => {
                if integer_range_contains(target, self) {
                    CastPolicy::Exact
                } else {
                    CastPolicy::Checked
                }
            }
            (DTypeKind::SignedInteger, DTypeKind::Float)
            | (DTypeKind::UnsignedInteger, DTypeKind::Float) => {
                if integer_is_exact_in_float(self, target) {
                    CastPolicy::Exact
                } else {
                    CastPolicy::Lossy
                }
            }
            (DTypeKind::Float, DTypeKind::SignedInteger)
            | (DTypeKind::Float, DTypeKind::UnsignedInteger) => CastPolicy::Lossy,
            (DTypeKind::Float, DTypeKind::Float) => {
                if target.itemsize() >= self.itemsize() {
                    CastPolicy::Exact
                } else {
                    CastPolicy::Lossy
                }
            }
            (DTypeKind::Bool, _) => CastPolicy::Exact,
            (_, DTypeKind::Bool) => CastPolicy::Lossy,
        }
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

    pub fn cast_policy_to(self, target: DType) -> CastPolicy {
        self.dtype().cast_policy_to(target)
    }

    pub fn cast(self, target: DType, mode: CastMode) -> Option<Self> {
        let policy = self.cast_policy_to(target);
        if !mode.permits(policy) {
            return None;
        }

        match target {
            DType::Bool => Some(Self::Bool(match self {
                Self::Bool(value) => value,
                Self::I8(value) => value != 0,
                Self::I16(value) => value != 0,
                Self::I32(value) => value != 0,
                Self::I64(value) => value != 0,
                Self::Isize(value) => value != 0,
                Self::U8(value) => value != 0,
                Self::U16(value) => value != 0,
                Self::U32(value) => value != 0,
                Self::U64(value) => value != 0,
                Self::Usize(value) => value != 0,
                Self::F32(value) => value != 0.0,
                Self::F64(value) => value != 0.0,
            })),
            DType::I8 => cast_to_signed(self, mode, i8::MIN as i128, i8::MAX as i128, |value| {
                Self::I8(value as i8)
            }),
            DType::I16 => cast_to_signed(self, mode, i16::MIN as i128, i16::MAX as i128, |value| {
                Self::I16(value as i16)
            }),
            DType::I32 => cast_to_signed(self, mode, i32::MIN as i128, i32::MAX as i128, |value| {
                Self::I32(value as i32)
            }),
            DType::I64 => cast_to_signed(self, mode, i64::MIN as i128, i64::MAX as i128, |value| {
                Self::I64(value as i64)
            }),
            DType::Isize => {
                cast_to_signed(self, mode, isize::MIN as i128, isize::MAX as i128, |value| {
                    Self::Isize(value as isize)
                })
            }
            DType::U8 => {
                cast_to_unsigned(self, mode, u8::MAX as u128, |value| Self::U8(value as u8))
            }
            DType::U16 => {
                cast_to_unsigned(self, mode, u16::MAX as u128, |value| Self::U16(value as u16))
            }
            DType::U32 => {
                cast_to_unsigned(self, mode, u32::MAX as u128, |value| Self::U32(value as u32))
            }
            DType::U64 => {
                cast_to_unsigned(self, mode, u64::MAX as u128, |value| Self::U64(value as u64))
            }
            DType::Usize => cast_to_unsigned(self, mode, usize::MAX as u128, |value| {
                Self::Usize(value as usize)
            }),
            DType::F32 => cast_to_f32(self),
            DType::F64 => cast_to_f64(self),
        }
    }
}

impl CastMode {
    pub fn permits(self, policy: CastPolicy) -> bool {
        match (self, policy) {
            (_, CastPolicy::Forbidden) => false,
            (Self::Checked, CastPolicy::Lossy) => false,
            _ => true,
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

    fn from_scalar_value(value: ScalarValue) -> Option<Self>
    where
        Self: Sized;
}

pub fn infer_scalar_dtype<T: RuntimeScalar>(value: T) -> DType {
    value.infer_dtype()
}

fn integer_range_contains(target: DType, source: DType) -> bool {
    match (target.integer_bounds(), source.integer_bounds()) {
        (Some(target_bounds), Some(source_bounds)) => {
            target_bounds.min <= source_bounds.min && target_bounds.max >= source_bounds.max
        }
        _ => false,
    }
}

fn integer_is_exact_in_float(source: DType, target: DType) -> bool {
    let mantissa_bits = match target {
        DType::F32 => 24,
        DType::F64 => 53,
        _ => return false,
    };

    if source.is_bool() {
        return true;
    }

    if source.is_signed_integer() {
        source.integer_bits().is_some_and(|bits| bits.saturating_sub(1) <= mantissa_bits)
    } else {
        source.integer_bits().is_some_and(|bits| bits <= mantissa_bits)
    }
}

#[derive(Debug, Clone, Copy)]
struct IntegerBounds {
    min: i128,
    max: i128,
}

impl DType {
    fn integer_bits(self) -> Option<usize> {
        match self.kind() {
            DTypeKind::Bool => Some(1),
            DTypeKind::SignedInteger | DTypeKind::UnsignedInteger => Some(self.itemsize() * 8),
            DTypeKind::Float => None,
        }
    }

    fn integer_bounds(self) -> Option<IntegerBounds> {
        match self {
            Self::Bool => Some(IntegerBounds { min: 0, max: 1 }),
            Self::I8 => Some(IntegerBounds { min: i8::MIN as i128, max: i8::MAX as i128 }),
            Self::I16 => Some(IntegerBounds { min: i16::MIN as i128, max: i16::MAX as i128 }),
            Self::I32 => Some(IntegerBounds { min: i32::MIN as i128, max: i32::MAX as i128 }),
            Self::I64 => Some(IntegerBounds { min: i64::MIN as i128, max: i64::MAX as i128 }),
            Self::Isize => Some(IntegerBounds { min: isize::MIN as i128, max: isize::MAX as i128 }),
            Self::U8 => Some(IntegerBounds { min: 0, max: u8::MAX as i128 }),
            Self::U16 => Some(IntegerBounds { min: 0, max: u16::MAX as i128 }),
            Self::U32 => Some(IntegerBounds { min: 0, max: u32::MAX as i128 }),
            Self::U64 => Some(IntegerBounds { min: 0, max: u64::MAX as i128 }),
            Self::Usize => Some(IntegerBounds { min: 0, max: usize::MAX as i128 }),
            Self::F32 | Self::F64 => None,
        }
    }
}

fn cast_to_signed(
    value: ScalarValue,
    mode: CastMode,
    min: i128,
    max: i128,
    ctor: fn(i128) -> ScalarValue,
) -> Option<ScalarValue> {
    let value = scalar_to_i128(value, mode)?;
    (min..=max).contains(&value).then(|| ctor(value))
}

fn cast_to_unsigned(
    value: ScalarValue,
    mode: CastMode,
    max: u128,
    ctor: fn(u128) -> ScalarValue,
) -> Option<ScalarValue> {
    let value = scalar_to_u128(value, mode)?;
    (value <= max).then(|| ctor(value))
}

fn cast_to_f32(value: ScalarValue) -> Option<ScalarValue> {
    let cast = match value {
        ScalarValue::Bool(value) => value as u8 as f32,
        ScalarValue::I8(value) => value as f32,
        ScalarValue::I16(value) => value as f32,
        ScalarValue::I32(value) => value as f32,
        ScalarValue::I64(value) => value as f32,
        ScalarValue::Isize(value) => value as f32,
        ScalarValue::U8(value) => value as f32,
        ScalarValue::U16(value) => value as f32,
        ScalarValue::U32(value) => value as f32,
        ScalarValue::U64(value) => value as f32,
        ScalarValue::Usize(value) => value as f32,
        ScalarValue::F32(value) => value,
        ScalarValue::F64(value) => {
            if value.is_nan() || value.is_infinite() {
                value as f32
            } else if (f32::MIN as f64..=f32::MAX as f64).contains(&value) {
                value as f32
            } else {
                return None;
            }
        }
    };

    Some(ScalarValue::F32(cast))
}

fn cast_to_f64(value: ScalarValue) -> Option<ScalarValue> {
    Some(ScalarValue::F64(match value {
        ScalarValue::Bool(value) => value as u8 as f64,
        ScalarValue::I8(value) => value as f64,
        ScalarValue::I16(value) => value as f64,
        ScalarValue::I32(value) => value as f64,
        ScalarValue::I64(value) => value as f64,
        ScalarValue::Isize(value) => value as f64,
        ScalarValue::U8(value) => value as f64,
        ScalarValue::U16(value) => value as f64,
        ScalarValue::U32(value) => value as f64,
        ScalarValue::U64(value) => value as f64,
        ScalarValue::Usize(value) => value as f64,
        ScalarValue::F32(value) => value as f64,
        ScalarValue::F64(value) => value,
    }))
}

fn scalar_to_i128(value: ScalarValue, mode: CastMode) -> Option<i128> {
    match value {
        ScalarValue::Bool(value) => Some(value as u8 as i128),
        ScalarValue::I8(value) => Some(value as i128),
        ScalarValue::I16(value) => Some(value as i128),
        ScalarValue::I32(value) => Some(value as i128),
        ScalarValue::I64(value) => Some(value as i128),
        ScalarValue::Isize(value) => Some(value as i128),
        ScalarValue::U8(value) => Some(value as i128),
        ScalarValue::U16(value) => Some(value as i128),
        ScalarValue::U32(value) => Some(value as i128),
        ScalarValue::U64(value) => Some(value as i128),
        ScalarValue::Usize(value) => Some(value as i128),
        ScalarValue::F32(value) if mode == CastMode::Lossy => float_to_i128(value as f64),
        ScalarValue::F64(value) if mode == CastMode::Lossy => float_to_i128(value),
        ScalarValue::F32(_) | ScalarValue::F64(_) => None,
    }
}

fn scalar_to_u128(value: ScalarValue, mode: CastMode) -> Option<u128> {
    match value {
        ScalarValue::Bool(value) => Some(value as u8 as u128),
        ScalarValue::I8(value) if value >= 0 => Some(value as u128),
        ScalarValue::I16(value) if value >= 0 => Some(value as u128),
        ScalarValue::I32(value) if value >= 0 => Some(value as u128),
        ScalarValue::I64(value) if value >= 0 => Some(value as u128),
        ScalarValue::Isize(value) if value >= 0 => Some(value as u128),
        ScalarValue::U8(value) => Some(value as u128),
        ScalarValue::U16(value) => Some(value as u128),
        ScalarValue::U32(value) => Some(value as u128),
        ScalarValue::U64(value) => Some(value as u128),
        ScalarValue::Usize(value) => Some(value as u128),
        ScalarValue::F32(value) if mode == CastMode::Lossy => float_to_u128(value as f64),
        ScalarValue::F64(value) if mode == CastMode::Lossy => float_to_u128(value),
        _ => None,
    }
}

fn float_to_i128(value: f64) -> Option<i128> {
    if !value.is_finite() {
        return None;
    }

    let truncated = value.trunc();
    (i128::MIN as f64..=i128::MAX as f64).contains(&truncated).then_some(truncated as i128)
}

fn float_to_u128(value: f64) -> Option<u128> {
    if !value.is_finite() || value < 0.0 {
        return None;
    }

    let truncated = value.trunc();
    (0.0..=u128::MAX as f64).contains(&truncated).then_some(truncated as u128)
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

                fn from_scalar_value(value: ScalarValue) -> Option<Self> {
                    match value {
                        ScalarValue::$variant(value) => Some(value),
                        _ => None,
                    }
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

    use super::{
        CastMode, CastPolicy, DType, DTypeKind, RuntimeDType, RuntimeScalar, ScalarValue,
        infer_scalar_dtype,
    };

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

    #[test]
    fn runtime_scalar_round_trips_through_scalar_value() {
        assert_eq!(i32::from_scalar_value(ScalarValue::I32(7)), Some(7));
        assert_eq!(u64::from_scalar_value(ScalarValue::U64(11)), Some(11));
        assert_eq!(f32::from_scalar_value(ScalarValue::F32(1.5)), Some(1.5));
        assert_eq!(i32::from_scalar_value(ScalarValue::U32(7)), None);
    }

    #[test]
    fn cast_policy_classifies_exact_checked_and_lossy_paths() {
        assert_eq!(DType::I8.cast_policy_to(DType::I16), CastPolicy::Exact);
        assert_eq!(DType::U32.cast_policy_to(DType::I16), CastPolicy::Checked);
        assert_eq!(DType::I64.cast_policy_to(DType::F32), CastPolicy::Lossy);
        assert_eq!(DType::F32.cast_policy_to(DType::F64), CastPolicy::Exact);
        assert_eq!(DType::F64.cast_policy_to(DType::I32), CastPolicy::Lossy);
        assert_eq!(DType::U8.cast_policy_to(DType::Bool), CastPolicy::Lossy);
    }

    #[test]
    fn cast_mode_only_permits_lossy_conversions_when_requested() {
        assert!(CastMode::Checked.permits(CastPolicy::Exact));
        assert!(CastMode::Checked.permits(CastPolicy::Checked));
        assert!(!CastMode::Checked.permits(CastPolicy::Lossy));
        assert!(CastMode::Lossy.permits(CastPolicy::Lossy));
        assert!(!CastMode::Lossy.permits(CastPolicy::Forbidden));
    }

    #[test]
    fn scalar_casts_support_exact_checked_and_lossy_paths() {
        assert_eq!(
            ScalarValue::I8(7).cast(DType::I16, CastMode::Checked),
            Some(ScalarValue::I16(7))
        );
        assert_eq!(ScalarValue::U16(255).cast(DType::I8, CastMode::Checked), None);
        assert_eq!(ScalarValue::I32(1).cast(DType::Bool, CastMode::Checked), None);
        assert_eq!(
            ScalarValue::I32(1).cast(DType::Bool, CastMode::Lossy),
            Some(ScalarValue::Bool(true))
        );
        assert_eq!(
            ScalarValue::F64(3.75).cast(DType::I32, CastMode::Lossy),
            Some(ScalarValue::I32(3))
        );
        assert_eq!(ScalarValue::F64(3.75).cast(DType::I32, CastMode::Checked), None);
        assert_eq!(ScalarValue::F64(f64::INFINITY).cast(DType::I32, CastMode::Lossy), None);
    }

    #[test]
    fn scalar_casts_reject_finite_overflow_on_narrow_float_targets() {
        assert_eq!(
            ScalarValue::F64((f32::MAX as f64) * 2.0).cast(DType::F32, CastMode::Lossy),
            None
        );
        assert_eq!(
            ScalarValue::F64(f64::INFINITY).cast(DType::F32, CastMode::Lossy),
            Some(ScalarValue::F32(f32::INFINITY))
        );
    }
}
