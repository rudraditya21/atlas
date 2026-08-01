use atlas_ndarray::{
    AtlasNdError,
    broadcast::{BroadcastMetadata, broadcast_pair, broadcast_shape, broadcast_strides},
};

#[test]
fn broadcast_shape_supports_trailing_dimension_alignment() {
    assert_eq!(
        broadcast_shape(&[8, 1, 6, 1], &[7, 1, 5]).unwrap(),
        vec![8, 7, 6, 5]
    );
}

#[test]
fn broadcast_shape_supports_scalar_expansion() {
    assert_eq!(broadcast_shape(&[], &[2, 3, 4]).unwrap(), vec![2, 3, 4]);
}

#[test]
fn broadcast_strides_mark_expanded_axes_with_zero_stride() {
    assert_eq!(
        broadcast_strides(&[1, 3], &[3, 1], &[2, 3]).unwrap(),
        vec![0, 1]
    );
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
