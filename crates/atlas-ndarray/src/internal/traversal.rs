use std::slice::Iter;

use super::{
    layout::{LayoutKind, PairLayoutKind, is_contiguous_layout, layout_kind, pair_layout_kind},
    shape::element_count,
};

pub(crate) fn value_iter<'a, T>(
    data: &'a [T],
    base_offset: usize,
    shape: &'a [usize],
    strides: &'a [usize],
) -> ValueIter<'a, T> {
    let len = element_count(shape);

    if len == 0 {
        return ValueIter::Empty;
    }

    if layout_kind(shape, strides) == LayoutKind::Contiguous {
        return ValueIter::Contiguous(contiguous_values(data, base_offset, len).iter());
    }

    ValueIter::Strided(StridedIter { data, shape, strides, base_offset, linear_index: 0, len })
}

/// Iterates maximal contiguous storage runs in logical row-major order.
pub(crate) fn logical_span_iter<'a, T>(
    data: &'a [T],
    base_offset: usize,
    shape: &'a [usize],
    strides: &'a [usize],
) -> LogicalSpanIter<'a, T> {
    debug_assert_eq!(shape.len(), strides.len());

    let len = element_count(shape);
    if len == 0 {
        return LogicalSpanIter {
            data,
            base_offset,
            shape,
            strides,
            outer_rank: 0,
            span_len: 0,
            next_span: 0,
            span_count: 0,
        };
    }

    let (outer_rank, span_len) = logical_span_shape(shape, strides);
    LogicalSpanIter {
        data,
        base_offset,
        shape,
        strides,
        outer_rank,
        span_len,
        next_span: 0,
        span_count: len / span_len,
    }
}

pub(crate) fn for_each_value<T, F>(
    data: &[T],
    base_offset: usize,
    shape: &[usize],
    strides: &[usize],
    mut f: F,
) where
    F: FnMut(&T),
{
    for value in value_iter(data, base_offset, shape, strides) {
        f(value);
    }
}

pub(crate) fn offset_iter<'a>(
    base_offset: usize,
    shape: &'a [usize],
    strides: &'a [usize],
) -> OffsetIter<'a> {
    let len = element_count(shape);

    if len == 0 {
        return OffsetIter::Empty;
    }

    if is_contiguous_layout(shape, strides) {
        return OffsetIter::Contiguous { next: base_offset, end: base_offset + len };
    }

    OffsetIter::Strided(StridedOffsetIter { shape, strides, base_offset, linear_index: 0, len })
}

pub(crate) fn try_for_each_value<T, E, F>(
    data: &[T],
    base_offset: usize,
    shape: &[usize],
    strides: &[usize],
    mut f: F,
) -> Result<(), E>
where
    F: FnMut(&T) -> Result<(), E>,
{
    for value in value_iter(data, base_offset, shape, strides) {
        f(value)?;
    }

    Ok(())
}

#[cfg(test)]
pub(crate) fn lane_value_iter<'a, T>(
    data: &'a [T],
    base_offset: usize,
    len: usize,
    stride: usize,
) -> LaneValueIter<'a, T> {
    if len == 0 {
        return LaneValueIter::Empty;
    }

    if stride == 1 {
        return LaneValueIter::Contiguous(contiguous_values(data, base_offset, len).iter());
    }

    LaneValueIter::Strided(StridedLaneIter { data, base_offset, stride, next: 0, len })
}

pub(crate) fn offset_pair_iter<'a>(
    lhs_base_offset: usize,
    rhs_base_offset: usize,
    shape: &'a [usize],
    lhs_strides: &'a [usize],
    rhs_strides: &'a [usize],
) -> OffsetPairIter<'a> {
    debug_assert_eq!(shape.len(), lhs_strides.len());
    debug_assert_eq!(shape.len(), rhs_strides.len());

    let len = element_count(shape);
    if len == 0 {
        return OffsetPairIter::Empty;
    }

    match pair_layout_kind(shape, lhs_strides, rhs_strides) {
        PairLayoutKind::Contiguous => OffsetPairIter::Contiguous {
            lhs_next: lhs_base_offset,
            rhs_next: rhs_base_offset,
            remaining: len,
        },
        PairLayoutKind::Broadcast => OffsetPairIter::Broadcast(broadcast_offset_pair_iter(
            lhs_base_offset,
            rhs_base_offset,
            shape,
            lhs_strides,
            rhs_strides,
        )),
        PairLayoutKind::Strided => OffsetPairIter::Strided(strided_offset_pair_iter(
            lhs_base_offset,
            rhs_base_offset,
            shape,
            lhs_strides,
            rhs_strides,
        )),
    }
}

pub(crate) fn broadcast_offset_pair_iter<'a>(
    lhs_base_offset: usize,
    rhs_base_offset: usize,
    shape: &'a [usize],
    lhs_strides: &'a [usize],
    rhs_strides: &'a [usize],
) -> BroadcastOffsetPairIter<'a> {
    BroadcastOffsetPairIter {
        lhs_base_offset,
        rhs_base_offset,
        shape,
        lhs_strides,
        rhs_strides,
        linear_index: 0,
        len: element_count(shape),
    }
}

pub(crate) fn strided_offset_pair_iter<'a>(
    lhs_base_offset: usize,
    rhs_base_offset: usize,
    shape: &'a [usize],
    lhs_strides: &'a [usize],
    rhs_strides: &'a [usize],
) -> StridedOffsetPairIter<'a> {
    StridedOffsetPairIter {
        lhs_base_offset,
        rhs_base_offset,
        shape,
        lhs_strides,
        rhs_strides,
        linear_index: 0,
        len: element_count(shape),
    }
}

pub(crate) enum ValueIter<'a, T> {
    Empty,
    Contiguous(Iter<'a, T>),
    Strided(StridedIter<'a, T>),
}

pub(crate) struct LogicalSpanIter<'a, T> {
    data: &'a [T],
    base_offset: usize,
    shape: &'a [usize],
    strides: &'a [usize],
    outer_rank: usize,
    span_len: usize,
    next_span: usize,
    span_count: usize,
}

impl<'a, T> Iterator for LogicalSpanIter<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_span >= self.span_count {
            return None;
        }

        let offset = offset_from_linear_index(
            self.next_span,
            self.base_offset,
            &self.shape[..self.outer_rank],
            &self.strides[..self.outer_rank],
        );
        self.next_span += 1;

        Some(&self.data[offset..offset + self.span_len])
    }
}

impl<'a, T> Iterator for ValueIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Empty => None,
            Self::Contiguous(iter) => iter.next(),
            Self::Strided(iter) => iter.next(),
        }
    }
}

#[cfg(test)]
pub(crate) enum LaneValueIter<'a, T> {
    Empty,
    Contiguous(Iter<'a, T>),
    Strided(StridedLaneIter<'a, T>),
}

#[cfg(test)]
impl<'a, T> Iterator for LaneValueIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Empty => None,
            Self::Contiguous(iter) => iter.next(),
            Self::Strided(iter) => iter.next(),
        }
    }
}

pub(crate) enum OffsetIter<'a> {
    Empty,
    Contiguous { next: usize, end: usize },
    Strided(StridedOffsetIter<'a>),
}

impl Iterator for OffsetIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Empty => None,
            Self::Contiguous { next, end } => {
                if *next >= *end {
                    None
                } else {
                    let offset = *next;
                    *next += 1;
                    Some(offset)
                }
            }
            Self::Strided(iter) => iter.next(),
        }
    }
}

pub(crate) enum OffsetPairIter<'a> {
    Empty,
    Contiguous { lhs_next: usize, rhs_next: usize, remaining: usize },
    Broadcast(BroadcastOffsetPairIter<'a>),
    Strided(StridedOffsetPairIter<'a>),
}

impl Iterator for OffsetPairIter<'_> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Empty => None,
            Self::Contiguous { lhs_next, rhs_next, remaining } => {
                if *remaining == 0 {
                    None
                } else {
                    let current = (*lhs_next, *rhs_next);
                    *lhs_next += 1;
                    *rhs_next += 1;
                    *remaining -= 1;
                    Some(current)
                }
            }
            Self::Broadcast(iter) => iter.next(),
            Self::Strided(iter) => iter.next(),
        }
    }
}

pub(crate) struct StridedIter<'a, T> {
    data: &'a [T],
    shape: &'a [usize],
    strides: &'a [usize],
    base_offset: usize,
    linear_index: usize,
    len: usize,
}

impl<'a, T> Iterator for StridedIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.linear_index >= self.len {
            return None;
        }

        let offset =
            offset_from_linear_index(self.linear_index, self.base_offset, self.shape, self.strides);
        self.linear_index += 1;

        Some(&self.data[offset])
    }
}

#[cfg(test)]
pub(crate) struct StridedLaneIter<'a, T> {
    data: &'a [T],
    base_offset: usize,
    stride: usize,
    next: usize,
    len: usize,
}

#[cfg(test)]
impl<'a, T> Iterator for StridedLaneIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.len {
            return None;
        }

        let offset = self.base_offset + self.next * self.stride;
        self.next += 1;

        Some(&self.data[offset])
    }
}

pub(crate) struct StridedOffsetIter<'a> {
    shape: &'a [usize],
    strides: &'a [usize],
    base_offset: usize,
    linear_index: usize,
    len: usize,
}

impl Iterator for StridedOffsetIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.linear_index >= self.len {
            return None;
        }

        let offset =
            offset_from_linear_index(self.linear_index, self.base_offset, self.shape, self.strides);
        self.linear_index += 1;

        Some(offset)
    }
}

pub(crate) struct StridedOffsetPairIter<'a> {
    lhs_base_offset: usize,
    rhs_base_offset: usize,
    shape: &'a [usize],
    lhs_strides: &'a [usize],
    rhs_strides: &'a [usize],
    linear_index: usize,
    len: usize,
}

impl Iterator for StridedOffsetPairIter<'_> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.linear_index >= self.len {
            return None;
        }

        let offset = pair_offsets_from_linear_index(
            self.linear_index,
            self.lhs_base_offset,
            self.rhs_base_offset,
            self.shape,
            self.lhs_strides,
            self.rhs_strides,
        );
        self.linear_index += 1;

        Some(offset)
    }
}

pub(crate) struct BroadcastOffsetPairIter<'a> {
    lhs_base_offset: usize,
    rhs_base_offset: usize,
    shape: &'a [usize],
    lhs_strides: &'a [usize],
    rhs_strides: &'a [usize],
    linear_index: usize,
    len: usize,
}

impl<'a> Iterator for BroadcastOffsetPairIter<'a> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.linear_index >= self.len {
            return None;
        }

        let current = self.linear_index;
        self.linear_index += 1;
        Some(broadcast_offsets(
            current,
            self.lhs_base_offset,
            self.rhs_base_offset,
            self.shape,
            self.lhs_strides,
            self.rhs_strides,
        ))
    }
}

fn broadcast_offsets(
    linear_index: usize,
    lhs_base_offset: usize,
    rhs_base_offset: usize,
    shape: &[usize],
    lhs_strides: &[usize],
    rhs_strides: &[usize],
) -> (usize, usize) {
    let mut lhs_offset = lhs_base_offset;
    let mut rhs_offset = rhs_base_offset;

    for_each_coordinate(linear_index, shape, |axis, coordinate| {
        lhs_offset += coordinate * lhs_strides[axis];
        rhs_offset += coordinate * rhs_strides[axis];
    });

    (lhs_offset, rhs_offset)
}

fn offset_from_linear_index(
    linear_index: usize,
    base_offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> usize {
    debug_assert_eq!(shape.len(), strides.len());

    let mut offset = base_offset;

    for_each_coordinate(linear_index, shape, |axis, coordinate| {
        offset += coordinate * strides[axis];
    });

    offset
}

fn pair_offsets_from_linear_index(
    linear_index: usize,
    lhs_base_offset: usize,
    rhs_base_offset: usize,
    shape: &[usize],
    lhs_strides: &[usize],
    rhs_strides: &[usize],
) -> (usize, usize) {
    let mut lhs_offset = lhs_base_offset;
    let mut rhs_offset = rhs_base_offset;

    for_each_coordinate(linear_index, shape, |axis, coordinate| {
        lhs_offset += coordinate * lhs_strides[axis];
        rhs_offset += coordinate * rhs_strides[axis];
    });

    (lhs_offset, rhs_offset)
}

fn contiguous_values<T>(data: &[T], base_offset: usize, len: usize) -> &[T] {
    &data[base_offset..base_offset + len]
}

fn logical_span_shape(shape: &[usize], strides: &[usize]) -> (usize, usize) {
    let mut outer_rank = shape.len();
    let mut span_len = 1usize;
    let mut expected_stride = 1usize;

    for axis in (0..shape.len()).rev() {
        if shape[axis] <= 1 {
            outer_rank = axis;
            continue;
        }
        if strides[axis] != expected_stride {
            break;
        }

        outer_rank = axis;
        span_len *= shape[axis];
        expected_stride *= shape[axis];
    }

    (outer_rank, span_len)
}

fn for_each_coordinate<F>(mut linear_index: usize, shape: &[usize], mut f: F)
where
    F: FnMut(usize, usize),
{
    for axis in (0..shape.len()).rev() {
        let dim = shape[axis];
        let coordinate = if dim == 0 { 0 } else { linear_index % dim };
        linear_index = linear_index.checked_div(dim).unwrap_or(0);
        f(axis, coordinate);
    }
}

#[cfg(test)]
mod tests {
    use super::{lane_value_iter, logical_span_iter, offset_iter, offset_pair_iter, value_iter};
    use crate::{NDArray, OperandMetadata, SliceRange};

    fn spans<O: OperandMetadata<i32> + ?Sized>(operand: &O) -> Vec<Vec<i32>> {
        logical_span_iter(operand.data(), operand.offset(), operand.shape(), operand.strides())
            .map(|span| span.to_vec())
            .collect()
    }

    #[test]
    fn logical_span_iter_preserves_logical_order_for_array_layouts() {
        let contiguous = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let transposed = contiguous.view().transpose();
        let padded_source =
            NDArray::from_shape_vec([2, 4], vec![0_i32, 1, 2, 3, 4, 5, 6, 7]).unwrap();
        let padded = padded_source.view().slice([0, 0], [2, 3]).unwrap();
        let sliced = contiguous.view().slice([1, 0], [1, 3]).unwrap();
        let stepped_source = NDArray::from_shape_vec([2, 6], (0_i32..12).collect()).unwrap();
        let stepped = stepped_source
            .view()
            .slice_ranges([SliceRange::full(), SliceRange::new(Some(0), Some(6), 2)])
            .unwrap();
        let scalar = NDArray::from_shape_vec([], vec![7_i32]).unwrap();
        let empty = NDArray::<i32>::from_shape_vec([2, 0, 3], Vec::new()).unwrap();

        assert_eq!(spans(&contiguous), vec![vec![0, 1, 2, 3, 4, 5]]);
        assert_eq!(spans(&transposed), vec![vec![0], vec![3], vec![1], vec![4], vec![2], vec![5]]);
        assert_eq!(spans(&padded), vec![vec![0, 1, 2], vec![4, 5, 6]]);
        assert_eq!(spans(&sliced), vec![vec![3, 4, 5]]);
        assert_eq!(spans(&stepped), vec![vec![0], vec![2], vec![4], vec![6], vec![8], vec![10]]);
        assert_eq!(spans(&scalar), vec![vec![7]]);
        assert!(spans(&empty).is_empty());
    }

    #[test]
    fn value_iter_uses_contiguous_path_when_layout_is_row_major() {
        let data = [1_i32, 2, 3, 4];
        let collected: Vec<_> = value_iter(&data, 0, &[2, 2], &[2, 1]).copied().collect();

        assert_eq!(collected, vec![1, 2, 3, 4]);
    }

    #[test]
    fn value_iter_walks_strided_layouts_in_logical_order() {
        let data = [0_i32, 1, 2, 3, 4, 5];
        let collected: Vec<_> = value_iter(&data, 0, &[3, 2], &[1, 3]).copied().collect();

        assert_eq!(collected, vec![0, 3, 1, 4, 2, 5]);
    }

    #[test]
    fn lane_value_iter_uses_contiguous_path_for_unit_stride() {
        let data = [1_i32, 2, 3, 4];
        let collected: Vec<_> = lane_value_iter(&data, 1, 2, 1).copied().collect();

        assert_eq!(collected, vec![2, 3]);
    }

    #[test]
    fn lane_value_iter_walks_strided_lanes() {
        let data = [0_i32, 1, 2, 3, 4, 5];
        let collected: Vec<_> = lane_value_iter(&data, 0, 3, 2).copied().collect();

        assert_eq!(collected, vec![0, 2, 4]);
    }

    #[test]
    fn offset_pair_iter_expands_singleton_axes_in_broadcast_layout() {
        let offsets: Vec<_> = offset_pair_iter(0, 0, &[2, 3], &[1, 0], &[0, 1]).collect();

        assert_eq!(offsets, vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]);
    }

    #[test]
    fn offset_pair_iter_uses_contiguous_offsets_for_dense_inputs() {
        let offsets: Vec<_> = offset_pair_iter(4, 10, &[3], &[1], &[1]).collect();

        assert_eq!(offsets, vec![(4, 10), (5, 11), (6, 12)]);
    }

    #[test]
    fn offset_pair_iter_walks_generic_strided_pairs_in_logical_order() {
        let offsets: Vec<_> = offset_pair_iter(0, 0, &[3, 2], &[1, 3], &[3, 1]).collect();

        assert_eq!(offsets, vec![(0, 0), (3, 1), (1, 3), (4, 4), (2, 6), (5, 7)]);
    }

    #[test]
    fn offset_iter_walks_strided_layouts_in_logical_order() {
        let offsets: Vec<_> = offset_iter(0, &[3, 2], &[1, 3]).collect();

        assert_eq!(offsets, vec![0, 3, 1, 4, 2, 5]);
    }
}
