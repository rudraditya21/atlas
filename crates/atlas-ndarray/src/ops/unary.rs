use crate::{
    ArrayElement, NDArray, Numeric, OperandMetadata,
    internal::{layout::is_contiguous_layout, simd, value_iter},
    view::ArrayView,
};

mod sealed {
    pub trait Neg {}
    pub trait Abs {}
    pub trait Sign {}
    pub trait Round {}
    pub trait FloatClassify {}

    macro_rules! impl_unary_types {
        ($trait:ident for $($ty:ty),+ $(,)?) => {
            $(impl $trait for $ty {})+
        };
    }

    impl_unary_types!(Neg for i8, i16, i32, i64, isize, f32, f64);
    impl_unary_types!(Abs for i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64);
    impl_unary_types!(Sign for i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64);
    impl_unary_types!(Round for i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64);
    impl_unary_types!(FloatClassify for bool, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64);
}

pub trait UnaryNeg: Numeric + sealed::Neg {
    fn unary_neg(self) -> Self;
}

pub trait UnaryAbs: Numeric + sealed::Abs {
    fn unary_abs(self) -> Self;
}

pub trait UnarySign: Numeric + sealed::Sign {
    fn unary_sign(self) -> Self;
}

pub trait UnaryRound: Numeric + sealed::Round {
    fn unary_round(self) -> Self;
}

pub trait FloatClassify: ArrayElement + sealed::FloatClassify {
    fn unary_isnan(self) -> bool;
    fn unary_isinf(self) -> bool;
    fn unary_isfinite(self) -> bool;
}

macro_rules! impl_signed_unary {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl UnaryNeg for $ty {
                fn unary_neg(self) -> Self { self.wrapping_neg() }
            }

            impl UnaryAbs for $ty {
                fn unary_abs(self) -> Self { self.wrapping_abs() }
            }

            impl UnarySign for $ty {
                fn unary_sign(self) -> Self { (self > 0) as $ty - (self < 0) as $ty }
            }

            impl UnaryRound for $ty {
                fn unary_round(self) -> Self { self }
            }
        )+
    };
}

macro_rules! impl_unsigned_unary {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl UnaryAbs for $ty {
                fn unary_abs(self) -> Self { self }
            }

            impl UnarySign for $ty {
                fn unary_sign(self) -> Self { (self != 0) as $ty }
            }

            impl UnaryRound for $ty {
                fn unary_round(self) -> Self { self }
            }
        )+
    };
}

macro_rules! impl_float_unary {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl UnaryNeg for $ty {
                fn unary_neg(self) -> Self { -self }
            }

            impl UnaryAbs for $ty {
                fn unary_abs(self) -> Self { self.abs() }
            }

            impl UnarySign for $ty {
                fn unary_sign(self) -> Self { self.signum() }
            }

            impl UnaryRound for $ty {
                fn unary_round(self) -> Self { self.round() }
            }
        )+
    };
}

impl_signed_unary!(i8, i16, i32, i64, isize);
impl_unsigned_unary!(u8, u16, u32, u64, usize);
impl_float_unary!(f32, f64);

macro_rules! impl_non_float_classify {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl FloatClassify for $ty {
                fn unary_isnan(self) -> bool { false }
                fn unary_isinf(self) -> bool { false }
                fn unary_isfinite(self) -> bool { true }
            }
        )+
    };
}

macro_rules! impl_float_classify {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl FloatClassify for $ty {
                fn unary_isnan(self) -> bool { self.is_nan() }
                fn unary_isinf(self) -> bool { self.is_infinite() }
                fn unary_isfinite(self) -> bool { self.is_finite() }
            }
        )+
    };
}

impl_non_float_classify!(bool, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);
impl_float_classify!(f32, f64);

fn map_unary<T, O, F>(operand: &O, op: F) -> NDArray<T>
where
    T: ArrayElement,
    O: OperandMetadata<T> + ?Sized,
    F: Fn(T) -> T,
{
    let data = if is_contiguous_layout(operand.shape(), operand.strides()) {
        operand
            .dense_slice()
            .expect("contiguous operands always expose a dense storage slice")
            .iter()
            .copied()
            .map(op)
            .collect()
    } else {
        value_iter(operand.data(), operand.offset(), operand.shape(), operand.strides())
            .copied()
            .map(op)
            .collect()
    };

    NDArray::from_row_major_parts(operand.shape().to_vec(), data)
        .expect("unary operations preserve ndarray invariants")
}

fn map_unary_numeric<T, O, F, C>(operand: &O, op: F, contiguous_op: C) -> NDArray<T>
where
    T: Numeric,
    O: OperandMetadata<T> + ?Sized,
    F: Fn(T) -> T,
    C: Fn(&[T], &mut [T]),
{
    let data = if is_contiguous_layout(operand.shape(), operand.strides()) {
        let values =
            operand.dense_slice().expect("contiguous operands always expose a dense storage slice");
        let mut data = vec![T::zero(); values.len()];
        contiguous_op(values, &mut data);
        data
    } else {
        value_iter(operand.data(), operand.offset(), operand.shape(), operand.strides())
            .copied()
            .map(op)
            .collect()
    };

    NDArray::from_row_major_parts(operand.shape().to_vec(), data)
        .expect("unary operations preserve ndarray invariants")
}

fn map_unary_bool<T, O, F>(operand: &O, op: F) -> NDArray<bool>
where
    T: ArrayElement,
    O: OperandMetadata<T> + ?Sized,
    F: Fn(T) -> bool,
{
    let data = if is_contiguous_layout(operand.shape(), operand.strides()) {
        operand
            .dense_slice()
            .expect("contiguous operands always expose a dense storage slice")
            .iter()
            .copied()
            .map(op)
            .collect()
    } else {
        value_iter(operand.data(), operand.offset(), operand.shape(), operand.strides())
            .copied()
            .map(op)
            .collect()
    };

    NDArray::from_row_major_parts(operand.shape().to_vec(), data)
        .expect("unary classification preserves ndarray invariants")
}

macro_rules! impl_unary_operations {
    ($operand:ty) => {
        impl<T: UnaryNeg> $operand {
            /// Returns the elementwise additive inverse; signed integer minima wrap unchanged.
            pub fn neg(&self) -> NDArray<T> {
                map_unary_numeric(self, UnaryNeg::unary_neg, simd::neg_contiguous)
            }
        }

        impl<T: UnaryAbs> $operand {
            /// Returns elementwise magnitudes; unsigned integers are unchanged and signed minima wrap unchanged.
            pub fn abs(&self) -> NDArray<T> {
                map_unary_numeric(self, UnaryAbs::unary_abs, simd::abs_contiguous)
            }
        }

        impl<T: UnarySign> $operand {
            /// Returns elementwise signs: `-1`, `0`, or `1` for integers and `signum` for floats.
            pub fn sign(&self) -> NDArray<T> {
                map_unary(self, UnarySign::unary_sign)
            }
        }

        impl<T: UnaryRound> $operand {
            /// Rounds floating values to the nearest integer with ties away from zero; integer values are unchanged.
            pub fn round(&self) -> NDArray<T> {
                map_unary(self, UnaryRound::unary_round)
            }
        }

        impl<T: FloatClassify> $operand {
            /// Returns whether each element is NaN; non-floating values are always `false`.
            pub fn isnan(&self) -> NDArray<bool> {
                map_unary_bool(self, FloatClassify::unary_isnan)
            }

            /// Returns whether each element is infinite; non-floating values are always `false`.
            pub fn isinf(&self) -> NDArray<bool> {
                map_unary_bool(self, FloatClassify::unary_isinf)
            }

            /// Returns whether each element is finite; non-floating values are always `true`.
            pub fn isfinite(&self) -> NDArray<bool> {
                map_unary_bool(self, FloatClassify::unary_isfinite)
            }
        }
    };
}

impl_unary_operations!(NDArray<T>);
impl_unary_operations!(ArrayView<'_, T>);
