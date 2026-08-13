use atlas_linalg::{AtlasLinalgError, DotOutput, dot, matmul, norm};
use atlas_ndarray::NDArray;

#[test]
fn dot_supports_vector_inputs() {
    let lhs = NDArray::from_shape_vec([4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let rhs = NDArray::from_shape_vec([4], vec![0.5_f64, 1.0, 1.5, 2.0]).unwrap();

    match dot(&lhs, &rhs).unwrap() {
        DotOutput::Scalar(value) => assert_eq!(value, 15.0),
        other => panic!("expected scalar dot output, got {other:?}"),
    }
}

#[test]
fn dot_supports_vector_matrix_matrix_vector_and_matrix_matrix() {
    let vector = NDArray::from_shape_vec([3], vec![1_i32, 2, 3]).unwrap();
    let matrix = NDArray::from_shape_vec([3, 2], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
    let left = NDArray::from_shape_vec([2, 3], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
    let right = NDArray::from_shape_vec([3, 2], vec![7_i32, 8, 9, 10, 11, 12]).unwrap();

    match dot(&vector, &matrix).unwrap() {
        DotOutput::Array(result) => {
            assert_eq!(result.shape(), &[2]);
            assert_eq!(result.data(), &[22, 28]);
        }
        other => panic!("expected array dot output, got {other:?}"),
    }

    match dot(&left, &vector).unwrap() {
        DotOutput::Array(result) => {
            assert_eq!(result.shape(), &[2]);
            assert_eq!(result.data(), &[14, 32]);
        }
        other => panic!("expected array dot output, got {other:?}"),
    }

    match dot(&left, &right).unwrap() {
        DotOutput::Array(result) => {
            assert_eq!(result.shape(), &[2, 2]);
            assert_eq!(result.data(), &[58, 64, 139, 154]);
        }
        other => panic!("expected array dot output, got {other:?}"),
    }
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
fn matmul_rejects_vector_vector_operands_in_v0_scope() {
    let lhs = NDArray::from_shape_vec([3], vec![1_i32, 2, 3]).unwrap();
    let rhs = NDArray::from_shape_vec([3], vec![4_i32, 5, 6]).unwrap();

    assert_eq!(
        matmul(&lhs, &rhs).unwrap_err(),
        AtlasLinalgError::InvalidOperandRank { op: "matmul", left: 1, right: 1 }
    );
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
    let lhs = NDArray::<i32>::zeros([2, 2, 2]).unwrap();
    let rhs = NDArray::<i32>::zeros([2, 2]).unwrap();

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
        AtlasLinalgError::InvalidOperandRank { op: "matmul", left: 1, right: 1 }
    );
}

#[test]
fn norm_supports_default_vector_and_matrix_cases() {
    let vector = NDArray::from_shape_vec([2], vec![3.0_f64, 4.0]).unwrap();
    let matrix = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();

    assert!((norm(&vector).unwrap() - 5.0).abs() <= 1.0e-10);
    assert!((norm(&matrix).unwrap() - 30.0_f64.sqrt()).abs() <= 1.0e-10);
}

#[test]
fn norm_reports_exact_rank_validation_errors() {
    let scalar = NDArray::from_shape_vec([], vec![7.0_f64]).unwrap();
    let tensor = NDArray::<f64>::zeros([1, 1, 1]).unwrap();

    assert_eq!(
        norm(&scalar).unwrap_err(),
        AtlasLinalgError::InvalidInputRank { op: "norm", expected: "a 1-D or 2-D array", rank: 0 }
    );
    assert_eq!(
        norm(&tensor).unwrap_err(),
        AtlasLinalgError::InvalidInputRank { op: "norm", expected: "a 1-D or 2-D array", rank: 3 }
    );
}
