use std::slice::Iter;

use crate::layout::{compute_strides, element_count};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayoutKind {
    Contiguous,
    Strided,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PairLayoutKind {
    Contiguous,
    Broadcast,
    Strided,
}

pub(crate) fn is_contiguous_layout(shape: &[usize], strides: &[usize]) -> bool {
    debug_assert_eq!(shape.len(), strides.len());
    strides == compute_strides(shape)
}

pub(crate) fn layout_kind(shape: &[usize], strides: &[usize]) -> LayoutKind {
    if is_contiguous_layout(shape, strides) { LayoutKind::Contiguous } else { LayoutKind::Strided }
}

pub(crate) fn pair_layout_kind(
    shape: &[usize],
    lhs_strides: &[usize],
    rhs_strides: &[usize],
) -> PairLayoutKind {
    debug_assert_eq!(shape.len(), lhs_strides.len());
    debug_assert_eq!(shape.len(), rhs_strides.len());

    if layout_kind(shape, lhs_strides) == LayoutKind::Contiguous
        && layout_kind(shape, rhs_strides) == LayoutKind::Contiguous
    {
        return PairLayoutKind::Contiguous;
    }

    if lhs_strides.contains(&0) || rhs_strides.contains(&0) {
        return PairLayoutKind::Broadcast;
    }

    PairLayoutKind::Strided
}

pub(crate) fn value_iter<'a, T>(
    data: &'a [T],
    base_offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> ValueIter<'a, T> {
    let len = element_count(shape);

    if len == 0 {
        return ValueIter::Empty;
    }

    if layout_kind(shape, strides) == LayoutKind::Contiguous {
        return ValueIter::Contiguous(contiguous_values(data, base_offset, len).iter());
    }

    ValueIter::Strided(StridedIter {
        data,
        shape: shape.to_vec(),
        strides: strides.to_vec(),
        base_offset,
        linear_index: 0,
        len,
    })
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

pub(crate) enum LaneValueIter<'a, T> {
    Empty,
    Contiguous(Iter<'a, T>),
    Strided(StridedLaneIter<'a, T>),
}

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
    shape: Vec<usize>,
    strides: Vec<usize>,
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

        let offset = offset_from_linear_index(
            self.linear_index,
            self.base_offset,
            &self.shape,
            &self.strides,
        );
        self.linear_index += 1;

        Some(&self.data[offset])
    }
}

pub(crate) struct StridedLaneIter<'a, T> {
    data: &'a [T],
    base_offset: usize,
    stride: usize,
    next: usize,
    len: usize,
}

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
    use super::{
        LayoutKind, PairLayoutKind, lane_value_iter, layout_kind, offset_iter, offset_pair_iter,
        pair_layout_kind, value_iter,
    };

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

    #[test]
    fn layout_kind_classifies_row_major_and_strided_layouts() {
        assert_eq!(layout_kind(&[2, 3], &[3, 1]), LayoutKind::Contiguous);
        assert_eq!(layout_kind(&[2, 3], &[1, 2]), LayoutKind::Strided);
    }

    #[test]
    fn pair_layout_kind_distinguishes_contiguous_broadcast_and_strided_cases() {
        assert_eq!(pair_layout_kind(&[2, 3], &[3, 1], &[3, 1]), PairLayoutKind::Contiguous);
        assert_eq!(pair_layout_kind(&[2, 3], &[3, 1], &[0, 1]), PairLayoutKind::Broadcast);
        assert_eq!(pair_layout_kind(&[2, 3], &[1, 2], &[3, 1]), PairLayoutKind::Strided);
    }
}
