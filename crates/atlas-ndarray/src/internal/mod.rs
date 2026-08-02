pub(crate) mod layout;
pub(crate) mod shape;
pub(crate) mod simd;
pub(crate) mod traversal;

use self::shape::{checked_compute_strides, checked_element_count};
pub(crate) use self::traversal::{
    OffsetIter, OffsetPairIter, ValueIter, broadcast_offset_pair_iter, for_each_value, offset_iter,
    offset_pair_iter, try_for_each_value, value_iter,
};
use crate::{AtlasNdError, AtlasNdResult};

pub(crate) fn validate_owned_array_invariants(
    data_len: usize,
    shape: &[usize],
    strides: &[usize],
) -> AtlasNdResult<()> {
    if shape.len() != strides.len() {
        return Err(AtlasNdError::InvalidShape);
    }

    let expected_len = checked_element_count(shape)?;
    if data_len != expected_len {
        return Err(AtlasNdError::ShapeMismatch { expected: expected_len, actual: data_len });
    }

    let expected_strides = checked_compute_strides(shape)?;
    if strides != expected_strides {
        return Err(AtlasNdError::InvalidShape);
    }

    Ok(())
}

pub(crate) fn validate_view_invariants(
    data_len: usize,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> AtlasNdResult<()> {
    if shape.len() != strides.len() {
        return Err(AtlasNdError::InvalidShape);
    }

    let len = checked_element_count(shape)?;
    if len == 0 {
        return if offset <= data_len { Ok(()) } else { Err(AtlasNdError::InvalidShape) };
    }

    let mut max_relative_offset = 0usize;
    for (&dim, &stride) in shape.iter().zip(strides.iter()) {
        let axis_extent = (dim - 1).checked_mul(stride).ok_or_else(|| {
            AtlasNdError::ShapeOverflow { op: "view validation", shape: shape.to_vec() }
        })?;
        max_relative_offset = max_relative_offset.checked_add(axis_extent).ok_or_else(|| {
            AtlasNdError::ShapeOverflow { op: "view validation", shape: shape.to_vec() }
        })?;
    }

    let max_offset = offset.checked_add(max_relative_offset).ok_or_else(|| {
        AtlasNdError::ShapeOverflow { op: "view validation", shape: shape.to_vec() }
    })?;

    if max_offset >= data_len {
        return Err(AtlasNdError::InvalidShape);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{validate_owned_array_invariants, validate_view_invariants};
    use crate::AtlasNdError;

    #[test]
    fn owned_array_invariant_validation_accepts_row_major_metadata() {
        assert_eq!(validate_owned_array_invariants(6, &[2, 3], &[3, 1]), Ok(()));
        assert_eq!(validate_owned_array_invariants(1, &[], &[]), Ok(()));
        assert_eq!(validate_owned_array_invariants(0, &[2, 0, 3], &[0, 3, 1]), Ok(()));
    }

    #[test]
    fn owned_array_invariant_validation_accepts_mixed_empty_shapes() {
        assert_eq!(validate_owned_array_invariants(0, &[0], &[1]), Ok(()));
        assert_eq!(validate_owned_array_invariants(0, &[0, 2, 0, 4], &[0, 0, 4, 1]), Ok(()));
        assert_eq!(validate_owned_array_invariants(0, &[3, 0, 0], &[0, 0, 1]), Ok(()));
    }

    #[test]
    fn owned_array_invariant_validation_rejects_invalid_metadata() {
        assert_eq!(
            validate_owned_array_invariants(6, &[2, 3], &[3]),
            Err(AtlasNdError::InvalidShape)
        );
        assert_eq!(
            validate_owned_array_invariants(5, &[2, 3], &[3, 1]),
            Err(AtlasNdError::ShapeMismatch { expected: 6, actual: 5 })
        );
        assert_eq!(
            validate_owned_array_invariants(6, &[2, 3], &[1, 3]),
            Err(AtlasNdError::InvalidShape)
        );
        assert_eq!(
            validate_owned_array_invariants(0, &[0, 2, 0, 4], &[0, 4, 0, 1]),
            Err(AtlasNdError::InvalidShape)
        );
        assert_eq!(
            validate_owned_array_invariants(0, &[], &[]),
            Err(AtlasNdError::ShapeMismatch { expected: 1, actual: 0 })
        );
    }

    #[test]
    fn view_invariant_validation_accepts_valid_metadata() {
        assert_eq!(validate_view_invariants(6, 0, &[2, 3], &[3, 1]), Ok(()));
        assert_eq!(validate_view_invariants(6, 0, &[3, 2], &[1, 3]), Ok(()));
        assert_eq!(validate_view_invariants(6, 1, &[2, 2], &[3, 1]), Ok(()));
        assert_eq!(validate_view_invariants(6, 0, &[0, 0], &[3, 1]), Ok(()));
        assert_eq!(validate_view_invariants(6, 3, &[1, 0], &[3, 1]), Ok(()));
    }

    #[test]
    fn view_invariant_validation_accepts_scalar_and_mixed_empty_boundaries() {
        assert_eq!(validate_view_invariants(1, 0, &[], &[]), Ok(()));
        assert_eq!(validate_view_invariants(6, 6, &[0], &[1]), Ok(()));
        assert_eq!(validate_view_invariants(6, 6, &[2, 0, 3], &[0, 3, 1]), Ok(()));
        assert_eq!(validate_view_invariants(6, 4, &[0, 2, 0], &[0, 0, 1]), Ok(()));
    }

    #[test]
    fn view_invariant_validation_rejects_invalid_metadata() {
        assert_eq!(validate_view_invariants(6, 0, &[2, 3], &[3]), Err(AtlasNdError::InvalidShape));
        assert_eq!(validate_view_invariants(6, 6, &[1], &[1]), Err(AtlasNdError::InvalidShape));
        assert_eq!(validate_view_invariants(6, 5, &[2], &[1]), Err(AtlasNdError::InvalidShape));
        assert_eq!(validate_view_invariants(6, 7, &[0], &[1]), Err(AtlasNdError::InvalidShape));
        assert_eq!(validate_view_invariants(1, 1, &[], &[]), Err(AtlasNdError::InvalidShape));
        assert_eq!(
            validate_view_invariants(6, 7, &[2, 0, 3], &[0, 3, 1]),
            Err(AtlasNdError::InvalidShape)
        );
    }
}
