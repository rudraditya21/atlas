use crate::{
    ArrayElement, AtlasNdResult, NDArray, internal::broadcast_offset_pair_iter,
    layout::broadcast::broadcast_pair,
};

mod sealed {
    pub trait Sealed {}

    macro_rules! impl_sealed {
        ($($ty:ty),+ $(,)?) => { $(impl Sealed for $ty {})+ };
    }

    impl_sealed!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);
}

pub trait BitwiseElement: ArrayElement + sealed::Sealed {
    fn bitand_elementwise(self, other: Self) -> Self;
    fn bitor_elementwise(self, other: Self) -> Self;
    fn bitxor_elementwise(self, other: Self) -> Self;
    fn bitnot_elementwise(self) -> Self;
}

macro_rules! impl_bitwise_element {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl BitwiseElement for $ty {
                fn bitand_elementwise(self, other: Self) -> Self { self & other }
                fn bitor_elementwise(self, other: Self) -> Self { self | other }
                fn bitxor_elementwise(self, other: Self) -> Self { self ^ other }
                fn bitnot_elementwise(self) -> Self { !self }
            }
        )+
    };
}

impl_bitwise_element!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

pub trait BitwiseOperand<T: BitwiseElement> {
    type Output;

    fn bitwise_apply<F>(self, lhs: &NDArray<T>, op: F) -> Self::Output
    where
        F: Fn(T, T) -> T + Copy;
}

impl<T: BitwiseElement> BitwiseOperand<T> for T {
    type Output = NDArray<T>;

    fn bitwise_apply<F>(self, lhs: &NDArray<T>, op: F) -> Self::Output
    where
        F: Fn(T, T) -> T + Copy,
    {
        map_bitwise_scalar(lhs, self, op)
    }
}

impl<T: BitwiseElement> BitwiseOperand<T> for &NDArray<T> {
    type Output = AtlasNdResult<NDArray<T>>;

    fn bitwise_apply<F>(self, lhs: &NDArray<T>, op: F) -> Self::Output
    where
        F: Fn(T, T) -> T + Copy,
    {
        map_bitwise_arrays(lhs, self, op)
    }
}

impl<T: BitwiseElement> NDArray<T> {
    pub fn bitand<Rhs: BitwiseOperand<T>>(&self, rhs: Rhs) -> Rhs::Output {
        rhs.bitwise_apply(self, |lhs, rhs| lhs.bitand_elementwise(rhs))
    }

    pub fn bitor<Rhs: BitwiseOperand<T>>(&self, rhs: Rhs) -> Rhs::Output {
        rhs.bitwise_apply(self, |lhs, rhs| lhs.bitor_elementwise(rhs))
    }

    pub fn bitxor<Rhs: BitwiseOperand<T>>(&self, rhs: Rhs) -> Rhs::Output {
        rhs.bitwise_apply(self, |lhs, rhs| lhs.bitxor_elementwise(rhs))
    }

    pub fn bitnot(&self) -> Self {
        NDArray::from_row_major_parts(
            self.shape().to_vec(),
            self.data().iter().copied().map(|value| value.bitnot_elementwise()).collect(),
        )
        .expect("bitwise not preserves ndarray invariants")
    }
}

fn map_bitwise_scalar<T, F>(lhs: &NDArray<T>, rhs: T, op: F) -> NDArray<T>
where
    T: BitwiseElement,
    F: Fn(T, T) -> T,
{
    NDArray::from_row_major_parts(
        lhs.shape().to_vec(),
        lhs.data().iter().copied().map(|lhs| op(lhs, rhs)).collect(),
    )
    .expect("bitwise scalar operations preserve ndarray invariants")
}

fn map_bitwise_arrays<T, F>(lhs: &NDArray<T>, rhs: &NDArray<T>, op: F) -> AtlasNdResult<NDArray<T>>
where
    T: BitwiseElement,
    F: Fn(T, T) -> T + Copy,
{
    if rhs.shape().is_empty() {
        return Ok(map_bitwise_scalar(lhs, rhs.data()[0], op));
    }

    if lhs.shape().is_empty() {
        return Ok(NDArray::from_row_major_parts(
            rhs.shape().to_vec(),
            rhs.data().iter().copied().map(|rhs| op(lhs.data()[0], rhs)).collect(),
        )
        .expect("bitwise scalar operations preserve ndarray invariants"));
    }

    let metadata = broadcast_pair(lhs.shape(), lhs.strides(), rhs.shape(), rhs.strides())?;
    let data = broadcast_offset_pair_iter(
        0,
        0,
        &metadata.shape,
        &metadata.lhs_strides,
        &metadata.rhs_strides,
    )
    .map(|(lhs_offset, rhs_offset)| op(lhs.data()[lhs_offset], rhs.data()[rhs_offset]))
    .collect();

    Ok(NDArray::from_row_major_parts(metadata.shape, data)
        .expect("bitwise array operations preserve ndarray invariants"))
}
