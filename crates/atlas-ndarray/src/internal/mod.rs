use std::slice::Iter;

use crate::layout::{compute_strides, element_count};

pub(crate) fn is_contiguous_layout(shape: &[usize], strides: &[usize]) -> bool {
    debug_assert_eq!(shape.len(), strides.len());
    strides == compute_strides(shape)
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

    if is_contiguous_layout(shape, strides) {
        let end = base_offset + len;
        return ValueIter::Contiguous(data[base_offset..end].iter());
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

pub(crate) fn broadcast_offset_pair_iter<'a>(
    shape: &'a [usize],
    lhs_strides: &'a [usize],
    rhs_strides: &'a [usize],
) -> BroadcastOffsetPairIter<'a> {
    debug_assert_eq!(shape.len(), lhs_strides.len());
    debug_assert_eq!(shape.len(), rhs_strides.len());

    BroadcastOffsetPairIter {
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

pub(crate) struct BroadcastOffsetPairIter<'a> {
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
        Some(broadcast_offsets(current, self.shape, self.lhs_strides, self.rhs_strides))
    }
}

fn broadcast_offsets(
    mut linear_index: usize,
    shape: &[usize],
    lhs_strides: &[usize],
    rhs_strides: &[usize],
) -> (usize, usize) {
    let mut lhs_offset = 0;
    let mut rhs_offset = 0;

    for axis in (0..shape.len()).rev() {
        let dim = shape[axis];
        let coordinate = if dim == 0 { 0 } else { linear_index % dim };
        linear_index = linear_index.checked_div(dim).unwrap_or(0);

        lhs_offset += coordinate * lhs_strides[axis];
        rhs_offset += coordinate * rhs_strides[axis];
    }

    (lhs_offset, rhs_offset)
}

fn offset_from_linear_index(
    mut linear_index: usize,
    base_offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> usize {
    debug_assert_eq!(shape.len(), strides.len());

    let mut offset = base_offset;

    for axis in (0..shape.len()).rev() {
        let dim = shape[axis];
        let coordinate = if dim == 0 { 0 } else { linear_index % dim };
        linear_index = linear_index.checked_div(dim).unwrap_or(0);
        offset += coordinate * strides[axis];
    }

    offset
}

#[cfg(test)]
mod tests {
    use super::{broadcast_offset_pair_iter, offset_iter, value_iter};

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
    fn broadcast_offset_pair_iter_expands_singleton_axes() {
        let offsets: Vec<_> = broadcast_offset_pair_iter(&[2, 3], &[1, 0], &[0, 1]).collect();

        assert_eq!(offsets, vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]);
    }

    #[test]
    fn offset_iter_walks_strided_layouts_in_logical_order() {
        let offsets: Vec<_> = offset_iter(0, &[3, 2], &[1, 3]).collect();

        assert_eq!(offsets, vec![0, 3, 1, 4, 2, 5]);
    }
}
