use atlas_ndarray::{
    AtlasNdError, BroadcastMetadata, broadcast_pair, broadcast_shape, broadcast_strides,
};

#[test]
fn broadcast_shape_supports_trailing_dimension_alignment() {
    assert_eq!(broadcast_shape(&[8, 1, 6, 1], &[7, 1, 5]).unwrap(), vec![8, 7, 6, 5]);
    assert_eq!(broadcast_shape(&[2, 1, 4], &[1, 3, 1]).unwrap(), vec![2, 3, 4]);
    assert_eq!(broadcast_shape(&[0, 4], &[1, 4]).unwrap(), vec![0, 4]);
}

#[test]
fn broadcast_shape_supports_scalar_expansion() {
    assert_eq!(broadcast_shape(&[], &[2, 3, 4]).unwrap(), vec![2, 3, 4]);
    assert_eq!(broadcast_shape(&[4, 1], &[]).unwrap(), vec![4, 1]);
}

#[test]
fn broadcast_strides_mark_expanded_axes_with_zero_stride() {
    assert_eq!(broadcast_strides(&[1, 3], &[3, 1], &[2, 3]).unwrap(), vec![0, 1]);
    assert_eq!(broadcast_strides(&[], &[], &[2, 3, 4]).unwrap(), vec![0, 0, 0]);
    assert_eq!(broadcast_strides(&[1, 0, 1], &[0, 1, 1], &[2, 0, 4]).unwrap(), vec![0, 1, 0]);
}

#[test]
fn broadcast_pair_returns_metadata_for_compatible_shapes() {
    let metadata = broadcast_pair(&[8, 1, 6, 1], &[6, 6, 1, 1], &[7, 1, 5], &[5, 5, 1]).unwrap();

    assert_eq!(
        metadata,
        BroadcastMetadata {
            shape: vec![8, 7, 6, 5],
            lhs_strides: vec![6, 0, 1, 0],
            rhs_strides: vec![0, 5, 0, 1],
        }
    );
}

#[test]
fn broadcast_pair_supports_scalar_and_zero_extent_operands() {
    assert_eq!(
        broadcast_pair(&[], &[], &[2, 3], &[3, 1]).unwrap(),
        BroadcastMetadata {
            shape: vec![2, 3],
            lhs_strides: vec![0, 0],
            rhs_strides: vec![3, 1],
        }
    );
    assert_eq!(
        broadcast_pair(&[2, 0, 4], &[0, 4, 1], &[1, 0, 1], &[0, 1, 1]).unwrap(),
        BroadcastMetadata {
            shape: vec![2, 0, 4],
            lhs_strides: vec![0, 4, 1],
            rhs_strides: vec![0, 1, 0],
        }
    );
}

#[test]
fn broadcast_shape_rejects_incompatible_shapes() {
    let error = broadcast_shape(&[2, 3], &[4, 3]).unwrap_err();

    assert_eq!(
        error,
        AtlasNdError::InvalidBroadcast {
            lhs: vec![2, 3],
            rhs: vec![4, 3],
            axis: 0,
            lhs_dim: 2,
            rhs_dim: 4,
        }
    );
}

#[test]
fn broadcast_rejections_cover_shape_and_stride_failures() {
    assert_eq!(
        broadcast_shape(&[2, 0], &[2, 3]).unwrap_err(),
        AtlasNdError::InvalidBroadcast {
            lhs: vec![2, 0],
            rhs: vec![2, 3],
            axis: 1,
            lhs_dim: 0,
            rhs_dim: 3,
        }
    );
    assert_eq!(broadcast_strides(&[2, 3], &[3], &[2, 3]).unwrap_err(), AtlasNdError::InvalidShape);
    assert_eq!(
        broadcast_strides(&[2, 3, 4], &[12, 4, 1], &[3, 4]).unwrap_err(),
        AtlasNdError::InvalidBroadcast {
            lhs: vec![2, 3, 4],
            rhs: vec![3, 4],
            axis: 0,
            lhs_dim: 2,
            rhs_dim: 1,
        }
    );
}
