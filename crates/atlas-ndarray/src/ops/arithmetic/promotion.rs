use crate::{CastMode, NDArray, Numeric, RuntimeScalar};

pub trait ArithmeticPromote<Rhs>: RuntimeScalar
where
    Rhs: RuntimeScalar,
{
    type Output: Numeric + RuntimeScalar;
}

mod sealed {
    pub(crate) trait Sealed {}

    impl Sealed for i8 {}
    impl Sealed for i16 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for isize {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for usize {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

pub(super) trait ArithmeticScalar: RuntimeScalar + sealed::Sealed {}

impl ArithmeticScalar for i8 {}
impl ArithmeticScalar for i16 {}
impl ArithmeticScalar for i32 {}
impl ArithmeticScalar for i64 {}
impl ArithmeticScalar for isize {}
impl ArithmeticScalar for u8 {}
impl ArithmeticScalar for u16 {}
impl ArithmeticScalar for u32 {}
impl ArithmeticScalar for u64 {}
impl ArithmeticScalar for usize {}
impl ArithmeticScalar for f32 {}
impl ArithmeticScalar for f64 {}

pub(super) fn cast_array_for_promotion<T, U>(array: &NDArray<T>) -> NDArray<U>
where
    T: Numeric + RuntimeScalar,
    U: Numeric + RuntimeScalar,
{
    array
        .astype_with_mode::<U>(CastMode::Lossy)
        .expect("promotion target must accept arithmetic operand values")
}

pub(super) fn cast_scalar_for_promotion<T, U>(scalar: T) -> U
where
    T: ArithmeticScalar,
    U: Numeric + RuntimeScalar,
{
    scalar
        .into_scalar_value()
        .cast(U::dtype(), CastMode::Lossy)
        .and_then(U::from_scalar_value)
        .expect("promotion target must accept arithmetic scalar values")
}

macro_rules! impl_promote_row {
    ($lhs:ty => { $($rhs:ty => $out:ty),+ $(,)? }) => {
        $(
            impl ArithmeticPromote<$rhs> for $lhs {
                type Output = $out;
            }
        )+
    };
}

impl_promote_row!(i8 => {
    i8 => i8,
    i16 => i16,
    i32 => i32,
    i64 => i64,
    u8 => i16,
    u16 => i32,
    u32 => i64,
    u64 => f64,
    f32 => f32,
    f64 => f64
});

impl_promote_row!(i16 => {
    i8 => i16,
    i16 => i16,
    i32 => i32,
    i64 => i64,
    u8 => i16,
    u16 => i32,
    u32 => i64,
    u64 => f64,
    f32 => f32,
    f64 => f64
});

impl_promote_row!(i32 => {
    i8 => i32,
    i16 => i32,
    i32 => i32,
    i64 => i64,
    u8 => i32,
    u16 => i32,
    u32 => i64,
    u64 => f64,
    f32 => f64,
    f64 => f64
});

impl_promote_row!(i64 => {
    i8 => i64,
    i16 => i64,
    i32 => i64,
    i64 => i64,
    u8 => i64,
    u16 => i64,
    u32 => i64,
    u64 => f64,
    f32 => f64,
    f64 => f64
});

impl_promote_row!(u8 => {
    i8 => i16,
    i16 => i16,
    i32 => i32,
    i64 => i64,
    u8 => u8,
    u16 => u16,
    u32 => u32,
    u64 => u64,
    f32 => f32,
    f64 => f64
});

impl_promote_row!(u16 => {
    i8 => i32,
    i16 => i32,
    i32 => i32,
    i64 => i64,
    u8 => u16,
    u16 => u16,
    u32 => u32,
    u64 => u64,
    f32 => f32,
    f64 => f64
});

impl_promote_row!(u32 => {
    i8 => i64,
    i16 => i64,
    i32 => i64,
    i64 => i64,
    u8 => u32,
    u16 => u32,
    u32 => u32,
    u64 => u64,
    f32 => f64,
    f64 => f64
});

impl_promote_row!(u64 => {
    i8 => f64,
    i16 => f64,
    i32 => f64,
    i64 => f64,
    u8 => u64,
    u16 => u64,
    u32 => u64,
    u64 => u64,
    f32 => f64,
    f64 => f64
});

impl_promote_row!(f32 => {
    i8 => f32,
    i16 => f32,
    i32 => f64,
    i64 => f64,
    u8 => f32,
    u16 => f32,
    u32 => f64,
    u64 => f64,
    f32 => f32,
    f64 => f64
});

impl_promote_row!(f64 => {
    i8 => f64,
    i16 => f64,
    i32 => f64,
    i64 => f64,
    u8 => f64,
    u16 => f64,
    u32 => f64,
    u64 => f64,
    f32 => f64,
    f64 => f64
});

#[cfg(target_pointer_width = "64")]
impl_promote_row!(isize => {
    i8 => isize,
    i16 => isize,
    i32 => isize,
    i64 => isize,
    isize => isize,
    u8 => isize,
    u16 => isize,
    u32 => isize,
    u64 => f64,
    usize => f64,
    f32 => f64,
    f64 => f64
});

#[cfg(target_pointer_width = "64")]
impl_promote_row!(usize => {
    i8 => f64,
    i16 => f64,
    i32 => f64,
    i64 => f64,
    isize => f64,
    u8 => usize,
    u16 => usize,
    u32 => usize,
    u64 => usize,
    usize => usize,
    f32 => f64,
    f64 => f64
});

#[cfg(target_pointer_width = "64")]
impl_promote_row!(i8 => { isize => isize, usize => f64 });
#[cfg(target_pointer_width = "64")]
impl_promote_row!(i16 => { isize => isize, usize => f64 });
#[cfg(target_pointer_width = "64")]
impl_promote_row!(i32 => { isize => isize, usize => f64 });
#[cfg(target_pointer_width = "64")]
impl_promote_row!(i64 => { isize => isize, usize => f64 });
#[cfg(target_pointer_width = "64")]
impl_promote_row!(u8 => { isize => isize, usize => usize });
#[cfg(target_pointer_width = "64")]
impl_promote_row!(u16 => { isize => isize, usize => usize });
#[cfg(target_pointer_width = "64")]
impl_promote_row!(u32 => { isize => isize, usize => usize });
#[cfg(target_pointer_width = "64")]
impl_promote_row!(u64 => { isize => f64, usize => usize });
#[cfg(target_pointer_width = "64")]
impl_promote_row!(f32 => { isize => f64, usize => f64 });
#[cfg(target_pointer_width = "64")]
impl_promote_row!(f64 => { isize => f64, usize => f64 });

#[cfg(target_pointer_width = "32")]
impl_promote_row!(isize => {
    i8 => isize,
    i16 => isize,
    i32 => isize,
    i64 => i64,
    isize => isize,
    u8 => isize,
    u16 => isize,
    u32 => i64,
    u64 => f64,
    usize => i64,
    f32 => f64,
    f64 => f64
});

#[cfg(target_pointer_width = "32")]
impl_promote_row!(usize => {
    i8 => i64,
    i16 => i64,
    i32 => i64,
    i64 => i64,
    isize => i64,
    u8 => usize,
    u16 => usize,
    u32 => usize,
    u64 => u64,
    usize => usize,
    f32 => f64,
    f64 => f64
});

#[cfg(target_pointer_width = "32")]
impl_promote_row!(i8 => { isize => isize, usize => i64 });
#[cfg(target_pointer_width = "32")]
impl_promote_row!(i16 => { isize => isize, usize => i64 });
#[cfg(target_pointer_width = "32")]
impl_promote_row!(i32 => { isize => isize, usize => i64 });
#[cfg(target_pointer_width = "32")]
impl_promote_row!(i64 => { isize => i64, usize => i64 });
#[cfg(target_pointer_width = "32")]
impl_promote_row!(u8 => { isize => isize, usize => usize });
#[cfg(target_pointer_width = "32")]
impl_promote_row!(u16 => { isize => isize, usize => usize });
#[cfg(target_pointer_width = "32")]
impl_promote_row!(u32 => { isize => i64, usize => usize });
#[cfg(target_pointer_width = "32")]
impl_promote_row!(u64 => { isize => f64, usize => u64 });
#[cfg(target_pointer_width = "32")]
impl_promote_row!(f32 => { isize => f64, usize => f64 });
#[cfg(target_pointer_width = "32")]
impl_promote_row!(f64 => { isize => f64, usize => f64 });
