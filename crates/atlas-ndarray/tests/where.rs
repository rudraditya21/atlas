use atlas_ndarray::{AtlasNdError, NDArray};

#[test]
fn where_select_supports_array_and_scalar_branch_combinations() {
    let condition = NDArray::from_shape_vec([2, 2], vec![true, false, true, false]).unwrap();
    let rhs = NDArray::from_shape_vec([2, 2], vec![1_i32, 2, 3, 4]).unwrap();

    let scalar_true = condition.r#where(9_i32, &rhs).unwrap();
    let scalar_false = condition.where_select(&rhs, 0_i32).unwrap();

    assert_eq!(scalar_true.shape(), &[2, 2]);
    assert_eq!(scalar_true.data(), &[9, 2, 9, 4]);
    assert_eq!(scalar_false.data(), &[1, 0, 3, 0]);
}

#[test]
fn where_select_broadcasts_scalar_conditions_and_array_branches() {
    let condition = NDArray::from_shape_vec([], vec![false]).unwrap();
    let x = NDArray::from_shape_vec([2, 1], vec![10_i32, 20]).unwrap();
    let y = NDArray::from_shape_vec([1, 3], vec![1_i32, 2, 3]).unwrap();

    let selected = condition.r#where(&x, &y).unwrap();

    assert_eq!(selected.shape(), &[2, 3]);
    assert_eq!(selected.data(), &[1, 2, 3, 1, 2, 3]);
}

#[test]
fn where_select_materializes_strided_condition_and_branch_views_in_logical_order() {
    let mask_source =
        NDArray::from_shape_vec([2, 3], vec![true, false, true, false, true, false]).unwrap();
    let x_source = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let y_source = NDArray::from_shape_vec([2, 3], vec![10_i32, 11, 12, 13, 14, 15]).unwrap();

    let condition = mask_source.view().transpose();
    let x = x_source.view().transpose();
    let y = y_source.view().transpose();
    let selected = condition.where_select(x, &y).unwrap();

    assert_eq!(selected.shape(), &[3, 2]);
    assert_eq!(selected.strides(), &[2, 1]);
    assert_eq!(selected.data(), &[0, 13, 11, 4, 2, 15]);
}

#[test]
fn where_select_rejects_incompatible_broadcast_shapes() {
    let condition = NDArray::from_shape_vec([2, 2], vec![true, false, true, false]).unwrap();
    let x = NDArray::from_shape_vec([3], vec![1_i32, 2, 3]).unwrap();
    let y = NDArray::from_shape_vec([2, 2], vec![4_i32, 5, 6, 7]).unwrap();

    assert_eq!(
        condition.r#where(&x, &y).unwrap_err(),
        AtlasNdError::InvalidBroadcast {
            lhs: vec![2, 2],
            rhs: vec![3],
            axis: 1,
            lhs_dim: 2,
            rhs_dim: 3,
        }
    );
}
