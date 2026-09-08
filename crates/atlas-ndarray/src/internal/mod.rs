pub(crate) mod layout;
pub(crate) mod materialize;
pub(crate) mod shape;
pub(crate) mod simd;
pub(crate) mod traversal;

use self::shape::{validate_row_major_shape_and_strides, validate_view_shape_and_strides};
pub(crate) use self::{
    materialize::materialize_contiguous_array,
    traversal::{
        ValueIter, broadcast_offset_pair_iter, for_each_value, offset_iter, offset_pair_iter,
        try_for_each_value, value_iter,
    },
};
use crate::{AtlasNdError, AtlasNdResult};

pub(crate) fn validate_owned_array_invariants(
    data_len: usize,
    shape: &[usize],
    strides: &[usize],
) -> AtlasNdResult<()> {
    let expected_len = validate_row_major_shape_and_strides(shape, strides)?;
    if data_len != expected_len {
        return Err(AtlasNdError::ShapeMismatch { expected: expected_len, actual: data_len });
    }

    Ok(())
}

pub(crate) fn validate_view_invariants(
    data_len: usize,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> AtlasNdResult<()> {
    validate_view_shape_and_strides(data_len, offset, shape, strides)
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
