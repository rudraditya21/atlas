use atlas_linalg::{AtlasLinalgError, dot, matmul};
use atlas_ndarray::NDArray;

#[test]
fn dot_supports_vector_inputs() {
    let lhs = NDArray::from_shape_vec([4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let rhs = NDArray::from_shape_vec([4], vec![0.5_f64, 1.0, 1.5, 2.0]).unwrap();

    assert_eq!(dot(&lhs, &rhs).unwrap(), 15.0);
}

#[test]
fn matmul_supports_vector_matrix_matrix_vector_and_matrix_matrix() {
    let vector = NDArray::from_shape_vec([3], vec![1_i32, 2, 3]).unwrap();
    let matrix = NDArray::from_shape_vec([3, 2], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
    let left = NDArray::from_shape_vec([2, 3], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
    let right = NDArray::from_shape_vec([3, 2], vec![7_i32, 8, 9, 10, 11, 12]).unwrap();

    assert_eq!(matmul(&vector, &matrix).unwrap().shape(), &[2]);
    assert_eq!(matmul(&vector, &matrix).unwrap().data(), &[22, 28]);

    assert_eq!(matmul(&left, &vector).unwrap().shape(), &[2]);
    assert_eq!(matmul(&left, &vector).unwrap().data(), &[14, 32]);

    assert_eq!(matmul(&left, &right).unwrap().shape(), &[2, 2]);
    assert_eq!(matmul(&left, &right).unwrap().data(), &[58, 64, 139, 154]);
}

#[test]
fn matmul_vector_vector_returns_scalar_shaped_array() {
    let lhs = NDArray::from_shape_vec([3], vec![1_i32, 2, 3]).unwrap();
    let rhs = NDArray::from_shape_vec([3], vec![4_i32, 5, 6]).unwrap();

    let result = matmul(&lhs, &rhs).unwrap();

    assert_eq!(result.shape(), &[] as &[usize]);
    assert_eq!(result.data(), &[32]);
}

#[test]
fn matmul_reports_exact_shape_mismatch_error() {
    let lhs = NDArray::from_shape_vec([2, 3], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
    let rhs = NDArray::from_shape_vec([4, 2], vec![1_i32, 2, 3, 4, 5, 6, 7, 8]).unwrap();

    assert_eq!(
        matmul(&lhs, &rhs).unwrap_err(),
        AtlasLinalgError::ShapeMismatch {
            op: "matmul",
            left: vec![2, 3],
            right: vec![4, 2],
            reason: "left matrix column count must match right matrix row count",
        }
    );
}

#[test]
fn matmul_rejects_ranks_above_two() {
    let lhs = NDArray::<i32>::zeros([2, 2, 2]);
    let rhs = NDArray::<i32>::zeros([2, 2]);

    assert_eq!(
        matmul(&lhs, &rhs).unwrap_err(),
        AtlasLinalgError::InvalidOperandRank { op: "matmul", left: 3, right: 2 }
    );
}

#[test]
fn dot_and_matmul_vector_vector_report_exact_mismatch_errors() {
    let lhs = NDArray::from_shape_vec([3], vec![1_i32, 2, 3]).unwrap();
    let rhs = NDArray::from_shape_vec([2], vec![4_i32, 5]).unwrap();

    assert_eq!(
        dot(&lhs, &rhs).unwrap_err(),
        AtlasLinalgError::ShapeMismatch {
            op: "dot",
            left: vec![3],
            right: vec![2],
            reason: "vector lengths must match",
        }
    );
    assert_eq!(
        matmul(&lhs, &rhs).unwrap_err(),
        AtlasLinalgError::ShapeMismatch {
            op: "matmul",
            left: vec![3],
            right: vec![2],
            reason: "vector lengths must match",
        }
    );
}
