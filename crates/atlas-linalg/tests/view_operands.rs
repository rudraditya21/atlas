use atlas_linalg::{AtlasLinalgError, DotOutput, dot, matmul, norm};
use atlas_ndarray::NDArray;

#[test]
fn dot_accepts_sliced_vector_views() {
    let lhs_base = NDArray::from_shape_vec([5], vec![1_i32, 2, 3, 4, 5]).unwrap();
    let rhs_base = NDArray::from_shape_vec([5], vec![5_i32, 4, 3, 2, 1]).unwrap();

    let lhs = lhs_base.view().slice([1], [3]).unwrap();
    let rhs = rhs_base.view().slice([1], [3]).unwrap();

    match dot(lhs, rhs).unwrap() {
        DotOutput::Scalar(value) => assert_eq!(value, 25),
        other => panic!("expected scalar dot output, got {other:?}"),
    }
}

#[test]
fn dot_accepts_matrix_vector_and_vector_matrix_views() {
    let vector = NDArray::from_shape_vec([3], vec![1_i32, 2, 3]).unwrap();
    let lhs_base = NDArray::from_shape_vec([2, 3], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
    let rhs_base = NDArray::from_shape_vec([2, 3], vec![1_i32, 3, 5, 2, 4, 6]).unwrap();

    match dot(lhs_base.view(), &vector).unwrap() {
        DotOutput::Array(result) => {
            assert_eq!(result.shape(), &[2]);
            assert_eq!(result.data(), &[14, 32]);
        }
        other => panic!("expected array dot output, got {other:?}"),
    }

    match dot(&vector, rhs_base.view().transpose()).unwrap() {
        DotOutput::Array(result) => {
            assert_eq!(result.shape(), &[2]);
            assert_eq!(result.data(), &[22, 28]);
        }
        other => panic!("expected array dot output, got {other:?}"),
    }
}

#[test]
fn matmul_accepts_transposed_view_operands() {
    let lhs = NDArray::from_shape_vec([2, 3], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
    let rhs_base = NDArray::from_shape_vec([2, 3], vec![7_i32, 9, 11, 8, 10, 12]).unwrap();

    assert_eq!(matmul(&lhs, rhs_base.view().transpose()).unwrap().data(), &[58, 64, 139, 154]);
}

#[test]
fn matmul_accepts_non_contiguous_sliced_views() {
    let lhs_base = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let rhs = NDArray::from_shape_vec([2], vec![10_i32, 20]).unwrap();

    let lhs = lhs_base.view().slice([0, 1], [2, 2]).unwrap();

    assert_eq!(matmul(lhs, &rhs).unwrap().data(), &[50, 140]);
}

#[test]
fn matmul_accepts_transposed_matrix_vector_views_without_materializing() {
    let lhs_base = NDArray::from_shape_vec([2, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs = NDArray::from_shape_vec([2], vec![10.0_f32, 20.0]).unwrap();

    assert_eq!(matmul(lhs_base.view().transpose(), &rhs).unwrap().data(), &[90.0, 120.0, 150.0]);
}

#[test]
fn dense_validation_errors_match_for_owned_and_view_vectors() {
    let lhs = NDArray::from_shape_vec([4], vec![1_i32, 2, 3, 4]).unwrap();
    let rhs = NDArray::from_shape_vec([3], vec![5_i32, 6, 7]).unwrap();
    let lhs_view = lhs.view().slice([0], [4]).unwrap();
    let rhs_view = rhs.view().slice([0], [3]).unwrap();

    assert_eq!(dot(&lhs, &rhs).unwrap_err(), dot(lhs_view.clone(), rhs_view.clone()).unwrap_err());
    assert_eq!(
        matmul(&lhs, &rhs).unwrap_err(),
        AtlasLinalgError::InvalidOperandRank { op: "matmul", left: 1, right: 1 }
    );
    assert_eq!(matmul(&lhs, &rhs).unwrap_err(), matmul(lhs_view, rhs_view).unwrap_err());
}

#[test]
fn norm_accepts_view_operands_without_materializing() {
    let vector_base = NDArray::from_shape_vec([4], vec![3.0_f64, 0.0, 4.0, 9.0]).unwrap();
    let matrix_base =
        NDArray::from_shape_vec([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let vector = vector_base.view().slice([0], [3]).unwrap();
    let matrix = matrix_base.view().transpose();

    assert!((norm(vector).unwrap() - 5.0).abs() <= 1.0e-10);
    assert!((norm(matrix).unwrap() - 91.0_f64.sqrt()).abs() <= 1.0e-10);
}
