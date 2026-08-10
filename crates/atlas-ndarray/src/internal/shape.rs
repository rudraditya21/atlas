use crate::{AtlasNdError, AtlasNdResult};

pub fn element_count(shape: &[usize]) -> usize {
    shape.iter().product()
}

pub fn checked_element_count(shape: &[usize]) -> AtlasNdResult<usize> {
    shape.iter().try_fold(1usize, |count, &dim| {
        count.checked_mul(dim).ok_or_else(|| AtlasNdError::ShapeOverflow {
            op: "element count",
            shape: shape.to_vec(),
        })
    })
}

pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }

    // Row-major contiguous layout: the last axis is unit-stride and each axis
    // to the left spans the full extent of the axis immediately to its right.
    let mut strides = vec![1usize; shape.len()];

    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    strides
}

pub fn checked_compute_strides(shape: &[usize]) -> AtlasNdResult<Vec<usize>> {
    if shape.is_empty() {
        return Ok(Vec::new());
    }

    let mut strides = vec![1usize; shape.len()];

    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1].checked_mul(shape[i + 1]).ok_or_else(|| {
            AtlasNdError::ShapeOverflow { op: "stride computation", shape: shape.to_vec() }
        })?;
    }

    Ok(strides)
}

pub(crate) fn checked_row_major_metadata(shape: &[usize]) -> AtlasNdResult<(usize, Vec<usize>)> {
    let len = checked_element_count(shape)?;
    let strides = checked_compute_strides(shape)?;

    Ok((len, strides))
}

pub(crate) fn validate_shape_and_strides(
    shape: &[usize],
    strides: &[usize],
) -> AtlasNdResult<usize> {
    if shape.len() != strides.len() {
        return Err(AtlasNdError::InvalidShape);
    }

    checked_element_count(shape)
}

pub(crate) fn validate_row_major_shape_and_strides(
    shape: &[usize],
    strides: &[usize],
) -> AtlasNdResult<usize> {
    let len = validate_shape_and_strides(shape, strides)?;
    let expected = checked_compute_strides(shape)?;

    if strides != expected {
        return Err(AtlasNdError::InvalidShape);
    }

    Ok(len)
}

pub(crate) fn validate_view_shape_and_strides(
    data_len: usize,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> AtlasNdResult<()> {
    let len = validate_shape_and_strides(shape, strides)?;

    if len == 0 {
        return if offset <= data_len { Ok(()) } else { Err(AtlasNdError::InvalidShape) };
    }

    let max_relative_offset = max_relative_offset(shape, strides)?;
    let max_offset = offset.checked_add(max_relative_offset).ok_or_else(|| {
        AtlasNdError::ShapeOverflow { op: "view validation", shape: shape.to_vec() }
    })?;

    if max_offset >= data_len {
        return Err(AtlasNdError::InvalidShape);
    }

    Ok(())
}

fn max_relative_offset(shape: &[usize], strides: &[usize]) -> AtlasNdResult<usize> {
    let mut max_relative_offset = 0usize;

    for (&dim, &stride) in shape.iter().zip(strides.iter()) {
        let axis_extent = dim.saturating_sub(1).checked_mul(stride).ok_or_else(|| {
            AtlasNdError::ShapeOverflow { op: "view validation", shape: shape.to_vec() }
        })?;
        max_relative_offset = max_relative_offset.checked_add(axis_extent).ok_or_else(|| {
            AtlasNdError::ShapeOverflow { op: "view validation", shape: shape.to_vec() }
        })?;
    }

    Ok(max_relative_offset)
}

#[cfg(test)]
mod tests {
    use crate::AtlasNdError;

    use super::{
        checked_compute_strides, checked_element_count, checked_row_major_metadata,
        compute_strides, element_count, validate_row_major_shape_and_strides,
        validate_shape_and_strides, validate_view_shape_and_strides,
    };

    #[test]
    fn element_count_handles_scalar_and_zero_sized_shapes() {
        assert_eq!(element_count(&[]), 1);
        assert_eq!(element_count(&[5]), 5);
        assert_eq!(element_count(&[2, 0, 4]), 0);
    }

    #[test]
    fn checked_element_count_handles_scalar_and_zero_sized_shapes() {
        assert_eq!(checked_element_count(&[]).unwrap(), 1);
        assert_eq!(checked_element_count(&[5]).unwrap(), 5);
        assert_eq!(checked_element_count(&[2, 0, 4]).unwrap(), 0);
    }

    #[test]
    fn checked_element_count_reports_overflow_explicitly() {
        assert_eq!(
            checked_element_count(&[usize::MAX, 2]).unwrap_err(),
            AtlasNdError::ShapeOverflow { op: "element count", shape: vec![usize::MAX, 2] }
        );
    }

    #[test]
    fn compute_strides_handles_scalar_shape() {
        assert_eq!(compute_strides(&[]), Vec::<usize>::new());
    }

    #[test]
    fn checked_compute_strides_handles_scalar_shape() {
        assert_eq!(checked_compute_strides(&[]).unwrap(), Vec::<usize>::new());
    }

    #[test]
    fn compute_strides_handles_one_dimensional_shape() {
        assert_eq!(compute_strides(&[5]), vec![1]);
    }

    #[test]
    fn checked_compute_strides_handles_one_dimensional_shape() {
        assert_eq!(checked_compute_strides(&[5]).unwrap(), vec![1]);
    }

    #[test]
    fn compute_strides_handles_two_dimensional_shape() {
        assert_eq!(compute_strides(&[2, 3]), vec![3, 1]);
    }

    #[test]
    fn checked_compute_strides_handles_two_dimensional_shape() {
        assert_eq!(checked_compute_strides(&[2, 3]).unwrap(), vec![3, 1]);
    }

    #[test]
    fn compute_strides_handles_three_dimensional_shape() {
        assert_eq!(compute_strides(&[2, 3, 4]), vec![12, 4, 1]);
    }

    #[test]
    fn checked_compute_strides_handles_three_dimensional_shape() {
        assert_eq!(checked_compute_strides(&[2, 3, 4]).unwrap(), vec![12, 4, 1]);
    }

    #[test]
    fn checked_compute_strides_reports_overflow_explicitly() {
        assert_eq!(
            checked_compute_strides(&[2, usize::MAX, 2]).unwrap_err(),
            AtlasNdError::ShapeOverflow { op: "stride computation", shape: vec![2, usize::MAX, 2] }
        );
    }

    #[test]
    fn checked_row_major_metadata_reuses_checked_shape_and_stride_computation() {
        assert_eq!(checked_row_major_metadata(&[2, 3]).unwrap(), (6, vec![3, 1]));
        assert_eq!(checked_row_major_metadata(&[]).unwrap(), (1, Vec::<usize>::new()));
        assert_eq!(checked_row_major_metadata(&[0]).unwrap(), (0, vec![1]));
        assert_eq!(checked_row_major_metadata(&[2, 0, 3]).unwrap(), (0, vec![0, 3, 1]));
    }

    #[test]
    fn validate_shape_and_strides_rejects_rank_mismatch() {
        assert_eq!(validate_shape_and_strides(&[2, 3], &[3]), Err(AtlasNdError::InvalidShape));
    }

    #[test]
    fn validate_row_major_shape_and_strides_checks_contiguous_layout() {
        assert_eq!(validate_row_major_shape_and_strides(&[2, 3], &[3, 1]), Ok(6));
        assert_eq!(
            validate_row_major_shape_and_strides(&[2, 3], &[1, 3]),
            Err(AtlasNdError::InvalidShape)
        );
    }

    #[test]
    fn validate_view_shape_and_strides_checks_bounds_and_empty_views() {
        assert_eq!(validate_view_shape_and_strides(6, 1, &[2, 2], &[3, 1]), Ok(()));
        assert_eq!(validate_view_shape_and_strides(6, 6, &[0], &[1]), Ok(()));
        assert_eq!(
            validate_view_shape_and_strides(6, 5, &[2], &[1]),
            Err(AtlasNdError::InvalidShape)
        );
    }
}
