use crate::{
    ArrayElement, NDArray, Numeric, OperandMetadata,
    internal::{layout::is_contiguous_layout, value_iter},
    view::ArrayView,
};

mod sealed {
    pub trait Neg {}
    pub trait Abs {}
    pub trait Sign {}
    pub trait Round {}

    macro_rules! impl_unary_types {
        ($trait:ident for $($ty:ty),+ $(,)?) => {
            $(impl $trait for $ty {})+
        };
    }

    impl_unary_types!(Neg for i8, i16, i32, i64, isize, f32, f64);
    impl_unary_types!(Abs for i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64);
    impl_unary_types!(Sign for i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64);
    impl_unary_types!(Round for i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64);
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

macro_rules! impl_unary_operations {
    ($operand:ty) => {
        impl<T: UnaryNeg> $operand {
            /// Returns the elementwise additive inverse; signed integer minima wrap unchanged.
            pub fn neg(&self) -> NDArray<T> {
                map_unary(self, UnaryNeg::unary_neg)
            }
        }

        impl<T: UnaryAbs> $operand {
            /// Returns elementwise magnitudes; unsigned integers are unchanged and signed minima wrap unchanged.
            pub fn abs(&self) -> NDArray<T> {
                map_unary(self, UnaryAbs::unary_abs)
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
    };
}

impl_unary_operations!(NDArray<T>);
impl_unary_operations!(ArrayView<'_, T>);
