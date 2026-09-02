use crate::{
    AtlasNdResult, NDArray, internal::broadcast_offset_pair_iter, layout::broadcast::broadcast_pair,
};

pub trait LogicalOperand {
    type Output;

    fn logical_apply<F>(self, lhs: &NDArray<bool>, op: F) -> Self::Output
    where
        F: Fn(bool, bool) -> bool + Copy;
}

impl LogicalOperand for bool {
    type Output = NDArray<bool>;

    fn logical_apply<F>(self, lhs: &NDArray<bool>, op: F) -> Self::Output
    where
        F: Fn(bool, bool) -> bool + Copy,
    {
        map_logical_scalar(lhs, self, op)
    }
}

impl LogicalOperand for &NDArray<bool> {
    type Output = AtlasNdResult<NDArray<bool>>;

    fn logical_apply<F>(self, lhs: &NDArray<bool>, op: F) -> Self::Output
    where
        F: Fn(bool, bool) -> bool + Copy,
    {
        map_logical_arrays(lhs, self, op)
    }
}

impl NDArray<bool> {
    pub fn logical_and<Rhs: LogicalOperand>(&self, rhs: Rhs) -> Rhs::Output {
        rhs.logical_apply(self, |lhs, rhs| lhs && rhs)
    }

    pub fn logical_or<Rhs: LogicalOperand>(&self, rhs: Rhs) -> Rhs::Output {
        rhs.logical_apply(self, |lhs, rhs| lhs || rhs)
    }

    pub fn logical_xor<Rhs: LogicalOperand>(&self, rhs: Rhs) -> Rhs::Output {
        rhs.logical_apply(self, |lhs, rhs| lhs ^ rhs)
    }

    pub fn logical_not(&self) -> Self {
        NDArray::from_row_major_parts(
            self.shape().to_vec(),
            self.data().iter().copied().map(|value| !value).collect(),
        )
        .expect("logical not preserves ndarray invariants")
    }
}

fn map_logical_scalar<F>(lhs: &NDArray<bool>, rhs: bool, op: F) -> NDArray<bool>
where
    F: Fn(bool, bool) -> bool,
{
    NDArray::from_row_major_parts(
        lhs.shape().to_vec(),
        lhs.data().iter().copied().map(|lhs| op(lhs, rhs)).collect(),
    )
    .expect("logical scalar operations preserve ndarray invariants")
}

fn map_logical_arrays<F>(
    lhs: &NDArray<bool>,
    rhs: &NDArray<bool>,
    op: F,
) -> AtlasNdResult<NDArray<bool>>
where
    F: Fn(bool, bool) -> bool + Copy,
{
    if rhs.shape().is_empty() {
        return Ok(map_logical_scalar(lhs, rhs.data()[0], op));
    }

    if lhs.shape().is_empty() {
        return Ok(NDArray::from_row_major_parts(
            rhs.shape().to_vec(),
            rhs.data().iter().copied().map(|rhs| op(lhs.data()[0], rhs)).collect(),
        )
        .expect("logical scalar operations preserve ndarray invariants"));
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
        .expect("logical array operations preserve ndarray invariants"))
}
