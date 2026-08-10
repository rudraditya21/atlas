use super::shape::element_count;

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
    if shape.is_empty() || element_count(shape) == 0 {
        return true;
    }

    let mut expected_stride = 1usize;

    for axis in (0..shape.len()).rev() {
        let dim = shape[axis];
        if dim <= 1 {
            continue;
        }

        if strides[axis] != expected_stride {
            return false;
        }

        expected_stride = expected_stride.saturating_mul(dim);
    }

    true
}

pub(crate) fn is_storage_dense_layout(shape: &[usize], strides: &[usize]) -> bool {
    debug_assert_eq!(shape.len(), strides.len());

    if shape.is_empty() || element_count(shape) == 0 {
        return true;
    }

    let mut axes: Vec<usize> = (0..shape.len()).filter(|&axis| shape[axis] > 1).collect();
    axes.sort_unstable_by_key(|&axis| strides[axis]);

    let mut expected_stride = 1usize;

    for axis in axes {
        if strides[axis] != expected_stride {
            return false;
        }

        expected_stride = expected_stride.saturating_mul(shape[axis]);
    }

    true
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

    if lhs_strides.contains(&0) || rhs_strides.contains(&0) {
        return PairLayoutKind::Broadcast;
    }

    if layout_kind(shape, lhs_strides) == LayoutKind::Contiguous
        && layout_kind(shape, rhs_strides) == LayoutKind::Contiguous
    {
        return PairLayoutKind::Contiguous;
    }

    PairLayoutKind::Strided
}

pub(crate) fn dense_storage_slice<'a, T>(
    data: &'a [T],
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> Option<&'a [T]> {
    if !is_storage_dense_layout(shape, strides) {
        return None;
    }

    let len = element_count(shape);
    if len == 0 {
        return Some(&data[offset..offset]);
    }

    Some(&data[offset..offset + len])
}

#[cfg(test)]
mod tests {
    use super::{
        LayoutKind, PairLayoutKind, dense_storage_slice, is_storage_dense_layout, layout_kind,
        pair_layout_kind,
    };

    #[test]
    fn layout_kind_classifies_row_major_and_strided_layouts() {
        assert_eq!(layout_kind(&[2, 3], &[3, 1]), LayoutKind::Contiguous);
        assert_eq!(layout_kind(&[2, 3], &[1, 2]), LayoutKind::Strided);
    }

    #[test]
    fn layout_kind_treats_empty_and_singleton_axes_as_contiguous_when_mapping_is_dense() {
        assert_eq!(layout_kind(&[0, 0], &[3, 1]), LayoutKind::Contiguous);
        assert_eq!(layout_kind(&[1, 3], &[6, 1]), LayoutKind::Contiguous);
        assert_eq!(layout_kind(&[2, 1, 3], &[3, 99, 1]), LayoutKind::Contiguous);
    }

    #[test]
    fn pair_layout_kind_distinguishes_contiguous_broadcast_and_strided_cases() {
        assert_eq!(pair_layout_kind(&[2, 3], &[3, 1], &[3, 1]), PairLayoutKind::Contiguous);
        assert_eq!(pair_layout_kind(&[2, 3], &[3, 1], &[0, 1]), PairLayoutKind::Broadcast);
        assert_eq!(pair_layout_kind(&[2, 3], &[1, 2], &[3, 1]), PairLayoutKind::Strided);
    }

    #[test]
    fn pair_layout_kind_keeps_zero_stride_empty_outputs_on_broadcast_path() {
        assert_eq!(pair_layout_kind(&[0, 3], &[3, 1], &[0, 0]), PairLayoutKind::Broadcast);
        assert_eq!(pair_layout_kind(&[2, 0, 4], &[0, 4, 1], &[0, 1, 0]), PairLayoutKind::Broadcast);
    }

    #[test]
    fn storage_dense_layout_accepts_transposed_dense_views() {
        assert!(is_storage_dense_layout(&[3, 2], &[1, 3]));
        assert!(is_storage_dense_layout(&[2, 3, 4], &[12, 1, 3]));
    }

    #[test]
    fn storage_dense_layout_ignores_singleton_axes_when_regions_are_dense() {
        assert!(is_storage_dense_layout(&[1, 3], &[6, 1]));
        assert!(is_storage_dense_layout(&[2, 1, 3], &[3, 99, 1]));
    }

    #[test]
    fn storage_dense_layout_rejects_gapped_slices() {
        assert!(!is_storage_dense_layout(&[2, 2], &[3, 1]));
        assert!(!is_storage_dense_layout(&[3], &[2]));
    }

    #[test]
    fn dense_storage_slice_returns_dense_regions_only() {
        let data = [0_i32, 1, 2, 3, 4, 5];

        assert_eq!(dense_storage_slice(&data, 0, &[2, 3], &[3, 1]), Some(&data[..]));
        assert_eq!(dense_storage_slice(&data, 0, &[3, 2], &[1, 3]), Some(&data[..]));
        assert_eq!(dense_storage_slice(&data, 3, &[1, 0], &[3, 1]), Some(&[] as &[i32]));
        assert_eq!(dense_storage_slice(&data, 1, &[2, 2], &[3, 1]), None);
    }
}
