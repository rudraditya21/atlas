use atlas_linalg::{dot, matmul};
use atlas_ndarray::NDArray;

#[test]
fn dot_accepts_sliced_vector_views() {
    let lhs_base = NDArray::from_shape_vec([5], vec![1_i32, 2, 3, 4, 5]).unwrap();
    let rhs_base = NDArray::from_shape_vec([5], vec![5_i32, 4, 3, 2, 1]).unwrap();

    let lhs = lhs_base.view().slice([1], [3]).unwrap();
    let rhs = rhs_base.view().slice([1], [3]).unwrap();

    assert_eq!(dot(lhs, rhs).unwrap(), 25);
}

#[test]
fn matmul_accepts_transposed_view_operands() {
    let lhs = NDArray::from_shape_vec([2, 3], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
    let rhs_base = NDArray::from_shape_vec([2, 3], vec![7_i32, 9, 11, 8, 10, 12]).unwrap();

    assert_eq!(
        matmul(&lhs, rhs_base.view().transpose()).unwrap().data(),
        &[58, 64, 139, 154]
    );
}

#[test]
fn matmul_accepts_non_contiguous_sliced_views() {
    let lhs_base = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let rhs = NDArray::from_shape_vec([2], vec![10_i32, 20]).unwrap();

    let lhs = lhs_base.view().slice([0, 1], [2, 2]).unwrap();

    assert_eq!(matmul(lhs, &rhs).unwrap().data(), &[50, 140]);
}
