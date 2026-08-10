use crate::{
    ArrayElement, AtlasNdResult, NDArray,
    internal::{
        broadcast_offset_pair_iter,
        layout::{PairLayoutKind, pair_layout_kind},
        offset_pair_iter,
    },
    layout::broadcast::{BroadcastMetadata, broadcast_pair},
};

fn from_bool_parts(shape: Vec<usize>, data: Vec<bool>) -> NDArray<bool> {
    NDArray::from_row_major_parts(shape, data)
        .expect("comparison kernels must preserve owned ndarray invariants")
}

fn compare_scalar<T, F>(lhs: &NDArray<T>, rhs: T, op: F) -> NDArray<bool>
where
    T: ArrayElement,
    F: Fn(T, T) -> bool + Copy,
{
    let data = lhs.data().iter().copied().map(|value| op(value, rhs)).collect();
    from_bool_parts(lhs.shape().to_vec(), data)
}

fn compare_contiguous<T, F>(lhs: &NDArray<T>, rhs: &NDArray<T>, op: F) -> NDArray<bool>
where
    T: ArrayElement,
    F: Fn(T, T) -> bool + Copy,
{
    let data = lhs
        .data()
        .iter()
        .copied()
        .zip(rhs.data().iter().copied())
        .map(|(left, right)| op(left, right))
        .collect();

    from_bool_parts(lhs.shape().to_vec(), data)
}

fn compare_broadcast<T, F>(
    lhs: &NDArray<T>,
    rhs: &NDArray<T>,
    metadata: BroadcastMetadata,
    op: F,
) -> NDArray<bool>
where
    T: ArrayElement,
    F: Fn(T, T) -> bool + Copy,
{
    let mut data = Vec::with_capacity(crate::element_count(&metadata.shape));

    for (lhs_offset, rhs_offset) in broadcast_offset_pair_iter(
        0,
        0,
        &metadata.shape,
        &metadata.lhs_strides,
        &metadata.rhs_strides,
    ) {
        data.push(op(lhs.data()[lhs_offset], rhs.data()[rhs_offset]));
    }

    from_bool_parts(metadata.shape, data)
}

fn compare_strided<T, F>(
    lhs: &NDArray<T>,
    rhs: &NDArray<T>,
    metadata: BroadcastMetadata,
    op: F,
) -> NDArray<bool>
where
    T: ArrayElement,
    F: Fn(T, T) -> bool + Copy,
{
    let mut data = Vec::with_capacity(crate::element_count(&metadata.shape));

    for (lhs_offset, rhs_offset) in
        offset_pair_iter(0, 0, &metadata.shape, &metadata.lhs_strides, &metadata.rhs_strides)
    {
        data.push(op(lhs.data()[lhs_offset], rhs.data()[rhs_offset]));
    }

    from_bool_parts(metadata.shape, data)
}

fn compare_arrays<T, F>(lhs: &NDArray<T>, rhs: &NDArray<T>, op: F) -> AtlasNdResult<NDArray<bool>>
where
    T: ArrayElement,
    F: Fn(T, T) -> bool + Copy,
{
    let metadata = broadcast_pair(lhs.shape(), lhs.strides(), rhs.shape(), rhs.strides())?;
    let layout_kind =
        pair_layout_kind(&metadata.shape, &metadata.lhs_strides, &metadata.rhs_strides);

    Ok(match layout_kind {
        PairLayoutKind::Contiguous => compare_contiguous(lhs, rhs, op),
        PairLayoutKind::Broadcast => compare_broadcast(lhs, rhs, metadata, op),
        PairLayoutKind::Strided => compare_strided(lhs, rhs, metadata, op),
    })
}

pub trait EqOperand<T: ArrayElement + PartialEq> {
    type Output;

    fn eq_to(self, lhs: &NDArray<T>) -> Self::Output;
}

pub trait NeOperand<T: ArrayElement + PartialEq> {
    type Output;

    fn ne_to(self, lhs: &NDArray<T>) -> Self::Output;
}

pub trait LtOperand<T: ArrayElement + PartialOrd> {
    type Output;

    fn lt_to(self, lhs: &NDArray<T>) -> Self::Output;
}

pub trait LeOperand<T: ArrayElement + PartialOrd> {
    type Output;

    fn le_to(self, lhs: &NDArray<T>) -> Self::Output;
}

pub trait GtOperand<T: ArrayElement + PartialOrd> {
    type Output;

    fn gt_to(self, lhs: &NDArray<T>) -> Self::Output;
}

pub trait GeOperand<T: ArrayElement + PartialOrd> {
    type Output;

    fn ge_to(self, lhs: &NDArray<T>) -> Self::Output;
}

impl<T: ArrayElement + PartialEq> EqOperand<T> for &NDArray<T> {
    type Output = AtlasNdResult<NDArray<bool>>;

    fn eq_to(self, lhs: &NDArray<T>) -> Self::Output {
        compare_arrays(lhs, self, |left, right| left == right)
    }
}

impl<T: ArrayElement + PartialEq> EqOperand<T> for T {
    type Output = NDArray<bool>;

    fn eq_to(self, lhs: &NDArray<T>) -> Self::Output {
        compare_scalar(lhs, self, |left, right| left == right)
    }
}

impl<T: ArrayElement + PartialEq> NeOperand<T> for &NDArray<T> {
    type Output = AtlasNdResult<NDArray<bool>>;

    fn ne_to(self, lhs: &NDArray<T>) -> Self::Output {
        compare_arrays(lhs, self, |left, right| left != right)
    }
}

impl<T: ArrayElement + PartialEq> NeOperand<T> for T {
    type Output = NDArray<bool>;

    fn ne_to(self, lhs: &NDArray<T>) -> Self::Output {
        compare_scalar(lhs, self, |left, right| left != right)
    }
}

impl<T: ArrayElement + PartialOrd> LtOperand<T> for &NDArray<T> {
    type Output = AtlasNdResult<NDArray<bool>>;

    fn lt_to(self, lhs: &NDArray<T>) -> Self::Output {
        compare_arrays(lhs, self, |left, right| left < right)
    }
}

impl<T: ArrayElement + PartialOrd> LtOperand<T> for T {
    type Output = NDArray<bool>;

    fn lt_to(self, lhs: &NDArray<T>) -> Self::Output {
        compare_scalar(lhs, self, |left, right| left < right)
    }
}

impl<T: ArrayElement + PartialOrd> LeOperand<T> for &NDArray<T> {
    type Output = AtlasNdResult<NDArray<bool>>;

    fn le_to(self, lhs: &NDArray<T>) -> Self::Output {
        compare_arrays(lhs, self, |left, right| left <= right)
    }
}

impl<T: ArrayElement + PartialOrd> LeOperand<T> for T {
    type Output = NDArray<bool>;

    fn le_to(self, lhs: &NDArray<T>) -> Self::Output {
        compare_scalar(lhs, self, |left, right| left <= right)
    }
}

impl<T: ArrayElement + PartialOrd> GtOperand<T> for &NDArray<T> {
    type Output = AtlasNdResult<NDArray<bool>>;

    fn gt_to(self, lhs: &NDArray<T>) -> Self::Output {
        compare_arrays(lhs, self, |left, right| left > right)
    }
}

impl<T: ArrayElement + PartialOrd> GtOperand<T> for T {
    type Output = NDArray<bool>;

    fn gt_to(self, lhs: &NDArray<T>) -> Self::Output {
        compare_scalar(lhs, self, |left, right| left > right)
    }
}

impl<T: ArrayElement + PartialOrd> GeOperand<T> for &NDArray<T> {
    type Output = AtlasNdResult<NDArray<bool>>;

    fn ge_to(self, lhs: &NDArray<T>) -> Self::Output {
        compare_arrays(lhs, self, |left, right| left >= right)
    }
}

impl<T: ArrayElement + PartialOrd> GeOperand<T> for T {
    type Output = NDArray<bool>;

    fn ge_to(self, lhs: &NDArray<T>) -> Self::Output {
        compare_scalar(lhs, self, |left, right| left >= right)
    }
}

impl<T: ArrayElement + PartialEq> NDArray<T> {
    pub fn eq<Rhs>(&self, rhs: Rhs) -> Rhs::Output
    where
        Rhs: EqOperand<T>,
    {
        rhs.eq_to(self)
    }

    pub fn ne<Rhs>(&self, rhs: Rhs) -> Rhs::Output
    where
        Rhs: NeOperand<T>,
    {
        rhs.ne_to(self)
    }
}

impl<T: ArrayElement + PartialOrd> NDArray<T> {
    pub fn lt<Rhs>(&self, rhs: Rhs) -> Rhs::Output
    where
        Rhs: LtOperand<T>,
    {
        rhs.lt_to(self)
    }

    pub fn le<Rhs>(&self, rhs: Rhs) -> Rhs::Output
    where
        Rhs: LeOperand<T>,
    {
        rhs.le_to(self)
    }

    pub fn gt<Rhs>(&self, rhs: Rhs) -> Rhs::Output
    where
        Rhs: GtOperand<T>,
    {
        rhs.gt_to(self)
    }

    pub fn ge<Rhs>(&self, rhs: Rhs) -> Rhs::Output
    where
        Rhs: GeOperand<T>,
    {
        rhs.ge_to(self)
    }
}
