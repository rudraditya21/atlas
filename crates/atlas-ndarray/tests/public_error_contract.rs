use atlas_ndarray::{AtlasNdError, NDArray, broadcast_shape};

#[test]
fn public_apis_report_stable_rank_shape_axis_slice_and_broadcast_errors() {
    let matrix = NDArray::from_shape_vec([2, 3], vec![0_i32; 6]).unwrap();
    let vector = NDArray::from_shape_vec([3], vec![0_i32; 3]).unwrap();

    assert_eq!(
        vector.diagonal(0).unwrap_err(),
        AtlasNdError::InvalidArgument { op: "diagonal", reason: "array must be two-dimensional" }
    );
    assert_eq!(
        NDArray::from_shape_vec([2, 2], vec![0_i32; 3]).unwrap_err(),
        AtlasNdError::ShapeMismatch { expected: 4, actual: 3 }
    );
    assert_eq!(matrix.sum_axis(-3).unwrap_err(), AtlasNdError::InvalidAxis { axis: -3, ndim: 2 });
    assert_eq!(
        matrix.view().slice([1, 2], [2, 2]).unwrap_err(),
        AtlasNdError::InvalidSlice { axis: 0, start: 1, len: 2, dim: 2 }
    );
    assert_eq!(
        broadcast_shape(&[2, 3], &[4, 3]).unwrap_err(),
        AtlasNdError::InvalidBroadcast {
            lhs: vec![2, 3],
            rhs: vec![4, 3],
            axis: 0,
            lhs_dim: 2,
            rhs_dim: 4,
        }
    );
}
