use atlas_linalg::{
    AtlasLinalgError, DotOutput, batched_dot, batched_transpose, dot, matmul, norm, trace,
};
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
fn batched_dot_supports_matching_batches_and_views() {
    let lhs = NDArray::from_shape_vec([2, 3], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
    let rhs = NDArray::from_shape_vec([2, 3], vec![7_i32, 8, 9, 10, 11, 12]).unwrap();
    let lhs_view = NDArray::from_shape_vec([2, 4], vec![1_i32, 2, 3, 0, 4, 5, 6, 0]).unwrap();
    let rhs_view = NDArray::from_shape_vec([2, 4], vec![7_i32, 8, 9, 0, 10, 11, 12, 0]).unwrap();

    assert_eq!(batched_dot(&lhs, &rhs).unwrap().data(), &[50, 167]);
    assert_eq!(
        batched_dot(
            lhs_view.view().slice([0, 0], [2, 3]).unwrap(),
            rhs_view.view().slice([0, 0], [2, 3]).unwrap(),
        )
        .unwrap()
        .data(),
        &[50, 167]
    );
}

#[test]
fn batched_dot_handles_empty_batches_and_zero_length_vectors() {
    let empty_batches = batched_dot(
        &NDArray::<i32>::zeros([0, 3]).unwrap(),
        &NDArray::<i32>::zeros([0, 3]).unwrap(),
    )
    .unwrap();
    let zero_length = batched_dot(
        &NDArray::<i32>::zeros([2, 0]).unwrap(),
        &NDArray::<i32>::zeros([2, 0]).unwrap(),
    )
    .unwrap();

    assert_eq!(empty_batches.shape(), &[0]);
    assert!(empty_batches.data().is_empty());
    assert_eq!(zero_length.shape(), &[2]);
    assert_eq!(zero_length.data(), &[0, 0]);
}

#[test]
fn batched_dot_rejects_mismatched_shapes() {
    let lhs = NDArray::<i32>::zeros([2, 3]).unwrap();
    let different_batches = NDArray::<i32>::zeros([1, 3]).unwrap();
    let different_lengths = NDArray::<i32>::zeros([2, 2]).unwrap();

    assert_eq!(
        batched_dot(&lhs, &different_batches).unwrap_err(),
        AtlasLinalgError::ShapeMismatch {
            op: "batched_dot",
            left: vec![2, 3],
            right: vec![1, 3],
            reason: "batch dimensions must match",
        }
    );
    assert_eq!(
        batched_dot(&lhs, &different_lengths).unwrap_err(),
        AtlasLinalgError::ShapeMismatch {
            op: "batched_dot",
            left: vec![2, 3],
            right: vec![2, 2],
            reason: "vector lengths must match",
        }
    );
}

#[test]
fn batched_transpose_swaps_only_matrix_axes() {
    let matrices =
        NDArray::from_shape_vec([2, 2, 3], vec![0_i32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).unwrap();

    let transposed = batched_transpose(&matrices).unwrap();

    assert_eq!(transposed.shape(), &[2, 3, 2]);
    assert_eq!(transposed.data(), &[0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11]);
}

#[test]
fn batched_transpose_supports_views() {
    let source = NDArray::from_shape_vec(
        [2, 2, 4],
        vec![1_i32, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0, 10, 11, 12, 0],
    )
    .unwrap();

    let transposed = batched_transpose(source.view().slice([0, 0, 0], [2, 2, 3]).unwrap()).unwrap();

    assert_eq!(transposed.shape(), &[2, 3, 2]);
    assert_eq!(transposed.data(), &[1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12]);
}

#[test]
fn batched_transpose_preserves_empty_dimensions() {
    let empty_batches = batched_transpose(&NDArray::<i32>::zeros([0, 2, 3]).unwrap()).unwrap();
    let zero_rows = batched_transpose(&NDArray::<i32>::zeros([2, 0, 3]).unwrap()).unwrap();
    let zero_columns = batched_transpose(&NDArray::<i32>::zeros([2, 3, 0]).unwrap()).unwrap();

    assert_eq!(empty_batches.shape(), &[0, 3, 2]);
    assert_eq!(zero_rows.shape(), &[2, 3, 0]);
    assert_eq!(zero_columns.shape(), &[2, 0, 3]);
    assert!(empty_batches.data().is_empty());
    assert!(zero_rows.data().is_empty());
    assert!(zero_columns.data().is_empty());
}

#[test]
fn batched_transpose_rejects_non_batched_matrices() {
    let matrix = NDArray::<i32>::zeros([2, 3]).unwrap();

    assert_eq!(
        batched_transpose(&matrix).unwrap_err(),
        AtlasLinalgError::InvalidInputRank {
            op: "batched_transpose",
            expected: "a rank-3 [batch, rows, columns] array",
            rank: 2,
        }
    );
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
fn matmul_supports_matching_batched_matrices() {
    let lhs = NDArray::from_shape_vec([2, 2, 3], vec![1_i32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        .unwrap();
    let rhs = NDArray::from_shape_vec([2, 3, 2], vec![1_i32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        .unwrap();

    let product = matmul(&lhs, &rhs).unwrap();

    assert_eq!(product.shape(), &[2, 2, 2]);
    assert_eq!(product.data(), &[22, 28, 49, 64, 220, 244, 301, 334]);
}

#[test]
fn batched_matmul_supports_strided_views() {
    let lhs = NDArray::from_shape_vec(
        [2, 2, 4],
        vec![1_i32, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0, 10, 11, 12, 0],
    )
    .unwrap();
    let rhs = NDArray::from_shape_vec(
        [2, 3, 3],
        vec![1_i32, 2, 0, 3, 4, 0, 5, 6, 0, 7, 8, 0, 9, 10, 0, 11, 12, 0],
    )
    .unwrap();

    let product = matmul(
        lhs.view().slice([0, 0, 0], [2, 2, 3]).unwrap(),
        rhs.view().slice([0, 0, 0], [2, 3, 2]).unwrap(),
    )
    .unwrap();

    assert_eq!(product.shape(), &[2, 2, 2]);
    assert_eq!(product.data(), &[22, 28, 49, 64, 220, 244, 301, 334]);
}

#[test]
fn batched_matrix_vector_matmul_supports_matching_batches() {
    let matrices =
        NDArray::from_shape_vec([2, 2, 3], vec![1_i32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            .unwrap();
    let vectors = NDArray::from_shape_vec([2, 3], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();

    let product = matmul(&matrices, &vectors).unwrap();

    assert_eq!(product.shape(), &[2, 2]);
    assert_eq!(product.data(), &[14, 32, 122, 167]);
}

#[test]
fn batched_matrix_vector_matmul_supports_strided_views() {
    let matrices = NDArray::from_shape_vec(
        [2, 2, 4],
        vec![1_i32, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0, 10, 11, 12, 0],
    )
    .unwrap();
    let vectors = NDArray::from_shape_vec([2, 4], vec![1_i32, 2, 3, 0, 4, 5, 6, 0]).unwrap();

    let product = matmul(
        matrices.view().slice([0, 0, 0], [2, 2, 3]).unwrap(),
        vectors.view().slice([0, 0], [2, 3]).unwrap(),
    )
    .unwrap();

    assert_eq!(product.shape(), &[2, 2]);
    assert_eq!(product.data(), &[14, 32, 122, 167]);
}

#[test]
fn batched_matrix_vector_matmul_handles_zero_sized_dimensions() {
    let empty_batches =
        matmul(&NDArray::<i32>::zeros([0, 2, 3]).unwrap(), &NDArray::<i32>::zeros([0, 3]).unwrap())
            .unwrap();
    let zero_inner =
        matmul(&NDArray::<i32>::zeros([2, 3, 0]).unwrap(), &NDArray::<i32>::zeros([2, 0]).unwrap())
            .unwrap();

    assert_eq!(empty_batches.shape(), &[0, 2]);
    assert!(empty_batches.data().is_empty());
    assert_eq!(zero_inner.shape(), &[2, 3]);
    assert_eq!(zero_inner.data(), &[0; 6]);
}

#[test]
fn batched_matrix_vector_matmul_rejects_incompatible_shapes() {
    let matrices = NDArray::<i32>::zeros([2, 2, 3]).unwrap();
    let different_batches = NDArray::<i32>::zeros([1, 3]).unwrap();
    let incompatible_vectors = NDArray::<i32>::zeros([2, 2]).unwrap();

    assert_eq!(
        matmul(&matrices, &different_batches).unwrap_err(),
        AtlasLinalgError::ShapeMismatch {
            op: "matmul",
            left: vec![2, 2, 3],
            right: vec![1, 3],
            reason: "batch dimensions must match",
        }
    );
    assert_eq!(
        matmul(&matrices, &incompatible_vectors).unwrap_err(),
        AtlasLinalgError::ShapeMismatch {
            op: "matmul",
            left: vec![2, 2, 3],
            right: vec![2, 2],
            reason: "left matrix column count must match vector length",
        }
    );
}

#[test]
fn batched_vector_matrix_matmul_supports_matching_batches() {
    let vectors = NDArray::from_shape_vec([2, 3], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
    let matrices =
        NDArray::from_shape_vec([2, 3, 2], vec![1_i32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            .unwrap();

    let product = matmul(&vectors, &matrices).unwrap();

    assert_eq!(product.shape(), &[2, 2]);
    assert_eq!(product.data(), &[22, 28, 139, 154]);
}

#[test]
fn batched_vector_matrix_matmul_supports_strided_views() {
    let vectors = NDArray::from_shape_vec([2, 4], vec![1_i32, 2, 3, 0, 4, 5, 6, 0]).unwrap();
    let matrices = NDArray::from_shape_vec(
        [2, 3, 3],
        vec![1_i32, 2, 0, 3, 4, 0, 5, 6, 0, 7, 8, 0, 9, 10, 0, 11, 12, 0],
    )
    .unwrap();

    let product = matmul(
        vectors.view().slice([0, 0], [2, 3]).unwrap(),
        matrices.view().slice([0, 0, 0], [2, 3, 2]).unwrap(),
    )
    .unwrap();

    assert_eq!(product.shape(), &[2, 2]);
    assert_eq!(product.data(), &[22, 28, 139, 154]);
}

#[test]
fn batched_vector_matrix_matmul_handles_zero_sized_dimensions() {
    let empty_batches =
        matmul(&NDArray::<i32>::zeros([0, 3]).unwrap(), &NDArray::<i32>::zeros([0, 3, 2]).unwrap())
            .unwrap();
    let zero_inner =
        matmul(&NDArray::<i32>::zeros([2, 0]).unwrap(), &NDArray::<i32>::zeros([2, 0, 3]).unwrap())
            .unwrap();
    let zero_columns =
        matmul(&NDArray::<i32>::zeros([2, 3]).unwrap(), &NDArray::<i32>::zeros([2, 3, 0]).unwrap())
            .unwrap();

    assert_eq!(empty_batches.shape(), &[0, 2]);
    assert!(empty_batches.data().is_empty());
    assert_eq!(zero_inner.shape(), &[2, 3]);
    assert_eq!(zero_inner.data(), &[0; 6]);
    assert_eq!(zero_columns.shape(), &[2, 0]);
    assert!(zero_columns.data().is_empty());
}

#[test]
fn batched_vector_matrix_matmul_rejects_incompatible_shapes() {
    let vectors = NDArray::<i32>::zeros([2, 3]).unwrap();
    let different_batches = NDArray::<i32>::zeros([1, 3, 2]).unwrap();
    let incompatible_matrices = NDArray::<i32>::zeros([2, 2, 2]).unwrap();

    assert_eq!(
        matmul(&vectors, &different_batches).unwrap_err(),
        AtlasLinalgError::ShapeMismatch {
            op: "matmul",
            left: vec![2, 3],
            right: vec![1, 3, 2],
            reason: "batch dimensions must match",
        }
    );
    assert_eq!(
        matmul(&vectors, &incompatible_matrices).unwrap_err(),
        AtlasLinalgError::ShapeMismatch {
            op: "matmul",
            left: vec![2, 3],
            right: vec![2, 2, 2],
            reason: "left vector length must match matrix row count",
        }
    );
}

#[test]
fn batched_matmul_preserves_empty_batches() {
    let lhs = NDArray::<i32>::zeros([0, 2, 3]).unwrap();
    let rhs = NDArray::<i32>::zeros([0, 3, 2]).unwrap();

    let product = matmul(&lhs, &rhs).unwrap();

    assert_eq!(product.shape(), &[0, 2, 2]);
    assert!(product.data().is_empty());
}

#[test]
fn batched_matmul_rejects_mismatched_batch_and_matrix_dimensions() {
    let lhs = NDArray::<i32>::zeros([2, 2, 3]).unwrap();
    let different_batches = NDArray::<i32>::zeros([1, 3, 2]).unwrap();
    let incompatible_matrices = NDArray::<i32>::zeros([2, 4, 2]).unwrap();

    assert_eq!(
        matmul(&lhs, &different_batches).unwrap_err(),
        AtlasLinalgError::ShapeMismatch {
            op: "matmul",
            left: vec![2, 2, 3],
            right: vec![1, 3, 2],
            reason: "batch dimensions must match",
        }
    );
    assert_eq!(
        matmul(&lhs, &incompatible_matrices).unwrap_err(),
        AtlasLinalgError::ShapeMismatch {
            op: "matmul",
            left: vec![2, 2, 3],
            right: vec![2, 4, 2],
            reason: "left matrix column count must match right matrix row count",
        }
    );
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
fn matmul_rejects_unsupported_rank_combinations() {
    let lhs = NDArray::<i32>::zeros([1, 2, 2, 2]).unwrap();
    let rhs = NDArray::<i32>::zeros([2, 2]).unwrap();

    assert_eq!(
        matmul(&lhs, &rhs).unwrap_err(),
        AtlasLinalgError::InvalidOperandRank { op: "matmul", left: 4, right: 2 }
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

#[test]
fn trace_supports_rank_two_owned_arrays() {
    let square = NDArray::from_shape_vec([2, 2], vec![1_i32, 2, 3, 4]).unwrap();
    let rectangular = NDArray::from_shape_vec([2, 3], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();

    assert_eq!(trace(&square).unwrap(), 5);
    assert_eq!(trace(&rectangular).unwrap(), 6);
}

#[test]
fn trace_reports_exact_rank_validation_errors() {
    let vector = NDArray::from_shape_vec([3], vec![1_i32, 2, 3]).unwrap();
    let scalar = NDArray::from_shape_vec([], vec![7_i32]).unwrap();

    assert_eq!(
        trace(&vector).unwrap_err(),
        AtlasLinalgError::InvalidInputRank { op: "trace", expected: "a 2-D array", rank: 1 }
    );
    assert_eq!(
        trace(&scalar).unwrap_err(),
        AtlasLinalgError::InvalidInputRank { op: "trace", expected: "a 2-D array", rank: 0 }
    );
}
