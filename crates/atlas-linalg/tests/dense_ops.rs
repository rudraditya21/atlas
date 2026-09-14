use atlas_linalg::{
    AtlasLinalgError, DotOutput, batched_diag, batched_dot, batched_transpose, conjugate_gradient,
    conjugate_gradient_with_diagnostics, dot, matmul, norm, solve_lower_triangular, solve_spd,
    solve_upper_triangular, symmetric_eigendecomposition, trace,
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
fn batched_triangular_solvers_handle_multiple_right_hand_sides_and_views() {
    let lower_factors = NDArray::from_shape_vec(
        [2, 2, 3],
        vec![2.0_f64, 0.0, 9.0, 3.0, 1.0, 9.0, 1.0, 0.0, 9.0, 2.0, 2.0, 9.0],
    )
    .unwrap();
    let lower_rhs = NDArray::from_shape_vec(
        [2, 2, 3],
        vec![4.0_f64, 2.0, 9.0, 5.0, 6.0, 9.0, 3.0, -1.0, 9.0, 14.0, 2.0, 9.0],
    )
    .unwrap();
    let upper =
        NDArray::from_shape_vec([2, 2, 2], vec![2.0_f64, 3.0, 0.0, 1.0, 1.0, 2.0, 0.0, 2.0])
            .unwrap();
    let upper_rhs =
        NDArray::from_shape_vec([2, 2, 2], vec![1.0_f64, 11.0, -1.0, 3.0, -1.0, 6.0, -4.0, 2.0])
            .unwrap();

    let lower_solution = solve_lower_triangular(
        lower_factors.view().slice([0, 0, 0], [2, 2, 2]).unwrap(),
        lower_rhs.view().slice([0, 0, 0], [2, 2, 2]).unwrap(),
    )
    .unwrap();
    let upper_solution = solve_upper_triangular(&upper, &upper_rhs).unwrap();

    assert_eq!(lower_solution.shape(), &[2, 2, 2]);
    assert_eq!(lower_solution.data(), &[2.0, 1.0, -1.0, 3.0, 3.0, -1.0, 4.0, 2.0]);
    assert_eq!(upper_solution.shape(), &[2, 2, 2]);
    assert_eq!(upper_solution.data(), &[2.0, 1.0, -1.0, 3.0, 3.0, 4.0, -2.0, 1.0]);
}

#[test]
fn batched_triangular_solvers_reject_invalid_factors_and_rhs_batches() {
    let non_square = NDArray::<f64>::zeros([2, 2, 3]).unwrap();
    let singular =
        NDArray::from_shape_vec([2, 2, 2], vec![1.0_f64, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
            .unwrap();
    let rhs = NDArray::<f64>::zeros([2, 2]).unwrap();
    let mismatched_batches = NDArray::<f64>::zeros([1, 2]).unwrap();

    assert!(matches!(
        solve_lower_triangular(&non_square, &rhs),
        Err(AtlasLinalgError::InvalidInputShape { op: "solve_lower_triangular", .. })
    ));
    assert_eq!(
        solve_upper_triangular(&singular, &rhs).unwrap_err(),
        AtlasLinalgError::SingularMatrix { op: "solve_upper_triangular", pivot: 1 }
    );
    assert_eq!(
        solve_lower_triangular(&singular, &mismatched_batches).unwrap_err(),
        AtlasLinalgError::ShapeMismatch {
            op: "solve_lower_triangular",
            left: vec![2, 2, 2],
            right: vec![1, 2],
            reason: "batch dimensions must match",
        }
    );
}

#[test]
fn batched_spd_solve_handles_multiple_right_hand_sides_and_views() {
    let matrices = NDArray::from_shape_vec(
        [2, 2, 3],
        vec![4.0_f64, 2.0, 9.0, 2.0, 3.0, 9.0, 2.0, 0.0, 9.0, 0.0, 5.0, 9.0],
    )
    .unwrap();
    let rhs = NDArray::from_shape_vec(
        [2, 2, 3],
        vec![8.0_f64, 2.0, 9.0, 8.0, 3.0, 9.0, 6.0, 8.0, 9.0, -5.0, 10.0, 9.0],
    )
    .unwrap();

    let solution = solve_spd(
        matrices.view().slice([0, 0, 0], [2, 2, 2]).unwrap(),
        rhs.view().slice([0, 0, 0], [2, 2, 2]).unwrap(),
    )
    .unwrap();

    assert_eq!(solution.shape(), &[2, 2, 2]);
    for (actual, expected) in solution.data().iter().zip([1.0, 0.0, 2.0, 1.0, 3.0, 4.0, -1.0, 2.0])
    {
        assert!((actual - expected).abs() < 1e-12);
    }
}

#[test]
fn batched_spd_solve_rejects_non_spd_batches_and_mismatched_shapes() {
    let valid =
        NDArray::from_shape_vec([2, 2, 2], vec![4.0_f64, 2.0, 2.0, 3.0, 2.0, 0.0, 0.0, 5.0])
            .unwrap();
    let non_spd =
        NDArray::from_shape_vec([2, 2, 2], vec![4.0_f64, 2.0, 2.0, 3.0, 1.0, 2.0, 2.0, 1.0])
            .unwrap();
    let rhs = NDArray::<f64>::zeros([2, 2]).unwrap();

    assert_eq!(
        solve_spd(&non_spd, &rhs).unwrap_err(),
        AtlasLinalgError::NotPositiveDefinite { op: "cholesky", index: 1 }
    );
    assert_eq!(
        solve_spd(&valid, &NDArray::<f64>::zeros([1, 2]).unwrap()).unwrap_err(),
        AtlasLinalgError::ShapeMismatch {
            op: "solve_spd",
            left: vec![2, 2, 2],
            right: vec![1, 2],
            reason: "batch dimensions must match",
        }
    );
    assert_eq!(
        solve_spd(&valid, &NDArray::<f64>::zeros([2, 1]).unwrap()).unwrap_err(),
        AtlasLinalgError::ShapeMismatch {
            op: "solve_spd",
            left: vec![2, 2, 2],
            right: vec![2, 1],
            reason: "right-hand side row count must match coefficient matrix row count",
        }
    );
}

#[test]
fn conjugate_gradient_solves_known_systems_and_logical_views() {
    let matrix =
        NDArray::from_shape_vec([3, 3], vec![4.0_f64, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0])
            .unwrap();
    let rhs = NDArray::from_shape_vec([3], vec![6.0_f64, 10.0, 8.0]).unwrap();
    let matrix_view =
        NDArray::from_shape_vec([2, 3], vec![4.0_f64, 1.0, 9.0, 1.0, 3.0, 9.0]).unwrap();
    let rhs_view = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 9.0]).unwrap();

    let solution = conjugate_gradient(&matrix, &rhs, 3, 1e-12).unwrap();
    let view_solution = conjugate_gradient(
        matrix_view.view().slice([0, 0], [2, 2]).unwrap(),
        rhs_view.view().slice([0], [2]).unwrap(),
        2,
        1e-12,
    )
    .unwrap();

    for (actual, expected) in solution.data().iter().zip([1.0, 2.0, 3.0]) {
        assert!((actual - expected).abs() < 1e-12);
    }
    for (actual, expected) in view_solution.data().iter().zip([1.0 / 11.0, 7.0 / 11.0]) {
        assert!((actual - expected).abs() < 1e-12);
    }
}

#[test]
fn conjugate_gradient_reports_iteration_limits_and_invalid_matrices() {
    let matrix =
        NDArray::from_shape_vec([3, 3], vec![4.0_f64, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0])
            .unwrap();
    let rhs = NDArray::from_shape_vec([3], vec![6.0_f64, 10.0, 8.0]).unwrap();
    let non_square = NDArray::<f64>::zeros([2, 3]).unwrap();
    let non_symmetric = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let non_spd = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 2.0, 2.0, 1.0]).unwrap();
    let short_rhs = NDArray::<f64>::zeros([2]).unwrap();
    let nonzero_rhs = NDArray::from_shape_vec([2], vec![1.0_f64, 0.0]).unwrap();

    assert_eq!(
        conjugate_gradient(&matrix, &rhs, 1, 1e-12).unwrap_err(),
        AtlasLinalgError::IterationLimit { op: "conjugate_gradient", iterations: 1 }
    );
    assert!(matches!(
        conjugate_gradient(&non_square, &short_rhs, 2, 1e-12),
        Err(AtlasLinalgError::InvalidInputShape { op: "conjugate_gradient", .. })
    ));
    assert!(matches!(
        conjugate_gradient(&non_symmetric, &short_rhs, 2, 1e-12),
        Err(AtlasLinalgError::InvalidInputShape { op: "conjugate_gradient", .. })
    ));
    assert!(matches!(
        conjugate_gradient(&non_spd, &nonzero_rhs, 2, 1e-12),
        Err(AtlasLinalgError::NotPositiveDefinite { op: "conjugate_gradient", .. })
    ));
}

#[test]
fn conjugate_gradient_diagnostics_report_converged_and_limited_outcomes() {
    let matrix =
        NDArray::from_shape_vec([3, 3], vec![4.0_f64, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0])
            .unwrap();
    let rhs = NDArray::from_shape_vec([3], vec![6.0_f64, 10.0, 8.0]).unwrap();

    let converged = conjugate_gradient_with_diagnostics(&matrix, &rhs, 3, 1e-12).unwrap();
    let limited = conjugate_gradient_with_diagnostics(&matrix, &rhs, 1, 1e-12).unwrap();

    assert!(converged.converged());
    assert_eq!(converged.iterations(), 3);
    assert!(converged.residual_norm() < 1e-12);
    assert_eq!(converged.solution().data().len(), 3);
    assert!(!limited.converged());
    assert_eq!(limited.iterations(), 1);
    assert!(limited.residual_norm() > 1e-12);
    assert_eq!(limited.solution().shape(), &[3]);
}

#[test]
fn symmetric_eigendecomposition_reconstructs_and_orthogonalizes() {
    let matrix = NDArray::from_shape_vec([2, 2], vec![4.0_f64, 1.0, 1.0, 3.0]).unwrap();

    let decomposition = symmetric_eigendecomposition(&matrix).unwrap();
    let eigenvalues = decomposition.eigenvalues();
    let eigenvectors = decomposition.eigenvectors();

    assert!((eigenvalues.data()[0] - (7.0 - 5.0_f64.sqrt()) / 2.0).abs() < 1e-12);
    assert!((eigenvalues.data()[1] - (7.0 + 5.0_f64.sqrt()) / 2.0).abs() < 1e-12);
    for row in 0..2 {
        for column in 0..2 {
            let reconstructed = (0..2)
                .map(|index| {
                    eigenvectors.data()[row * 2 + index]
                        * eigenvalues.data()[index]
                        * eigenvectors.data()[column * 2 + index]
                })
                .sum::<f64>();
            let orthogonality = (0..2)
                .map(|index| {
                    eigenvectors.data()[index * 2 + row] * eigenvectors.data()[index * 2 + column]
                })
                .sum::<f64>();

            assert!((reconstructed - matrix.data()[row * 2 + column]).abs() < 1e-12);
            assert!((orthogonality - if row == column { 1.0 } else { 0.0 }).abs() < 1e-12);
        }
    }
}

#[test]
fn symmetric_eigendecomposition_handles_repeated_values_and_views() {
    let repeated =
        NDArray::from_shape_vec([3, 3], vec![2.0_f64, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0])
            .unwrap();
    let view_source =
        NDArray::from_shape_vec([2, 3], vec![4.0_f64, 1.0, 9.0, 1.0, 3.0, 9.0]).unwrap();

    let repeated_decomposition = symmetric_eigendecomposition(&repeated).unwrap();
    let view_decomposition =
        symmetric_eigendecomposition(view_source.view().slice([0, 0], [2, 2]).unwrap()).unwrap();

    assert_eq!(repeated_decomposition.eigenvalues().data(), &[2.0, 2.0, 2.0]);
    for row in 0..3 {
        for column in 0..3 {
            let dot = (0..3)
                .map(|index| {
                    repeated_decomposition.eigenvectors().data()[index * 3 + row]
                        * repeated_decomposition.eigenvectors().data()[index * 3 + column]
                })
                .sum::<f64>();
            assert!((dot - if row == column { 1.0 } else { 0.0 }).abs() < 1e-12);
        }
    }
    assert!(
        (view_decomposition.eigenvalues().data()[0] - (7.0 - 5.0_f64.sqrt()) / 2.0).abs() < 1e-12
    );
    assert!(
        (view_decomposition.eigenvalues().data()[1] - (7.0 + 5.0_f64.sqrt()) / 2.0).abs() < 1e-12
    );
}

#[test]
fn symmetric_eigendecomposition_rejects_invalid_matrices() {
    let vector = NDArray::from_shape_vec([2], vec![1.0_f64, 2.0]).unwrap();
    let non_square = NDArray::<f64>::zeros([2, 3]).unwrap();
    let non_symmetric = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let non_finite = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 0.0, 0.0, f64::NAN]).unwrap();

    assert!(matches!(
        symmetric_eigendecomposition(&vector),
        Err(AtlasLinalgError::InvalidInputRank { op: "symmetric_eigendecomposition", .. })
    ));
    assert!(matches!(
        symmetric_eigendecomposition(&non_square),
        Err(AtlasLinalgError::InvalidInputShape { op: "symmetric_eigendecomposition", .. })
    ));
    assert!(matches!(
        symmetric_eigendecomposition(&non_symmetric),
        Err(AtlasLinalgError::InvalidInputShape { op: "symmetric_eigendecomposition", .. })
    ));
    assert!(matches!(
        symmetric_eigendecomposition(&non_finite),
        Err(AtlasLinalgError::NonFiniteInput { op: "symmetric_eigendecomposition" })
    ));
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
fn batched_diag_extracts_square_and_rectangular_diagonals() {
    let square = NDArray::from_shape_vec(
        [2, 3, 3],
        vec![0_i32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    )
    .unwrap();
    let rectangular =
        NDArray::from_shape_vec([2, 2, 3], vec![0_i32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).unwrap();

    assert_eq!(batched_diag(&square, 0).unwrap().data(), &[0, 4, 8, 9, 13, 17]);
    assert_eq!(batched_diag(&rectangular, 1).unwrap().data(), &[1, 5, 7, 11]);
    assert_eq!(batched_diag(&rectangular, -1).unwrap().data(), &[3, 9]);
}

#[test]
fn batched_diag_supports_views_and_empty_batches() {
    let source = NDArray::from_shape_vec(
        [2, 2, 4],
        vec![1_i32, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0, 10, 11, 12, 0],
    )
    .unwrap();
    let empty = NDArray::<i32>::zeros([0, 2, 3]).unwrap();

    let diagonal = batched_diag(source.view().slice([0, 0, 0], [2, 2, 3]).unwrap(), 0).unwrap();
    let empty_diagonal = batched_diag(&empty, 0).unwrap();

    assert_eq!(diagonal.shape(), &[2, 2]);
    assert_eq!(diagonal.data(), &[1, 5, 7, 11]);
    assert_eq!(empty_diagonal.shape(), &[0, 2]);
    assert!(empty_diagonal.data().is_empty());
}

#[test]
fn batched_diag_rejects_non_batched_matrices() {
    let matrix = NDArray::<i32>::zeros([2, 3]).unwrap();

    assert_eq!(
        batched_diag(&matrix, 0).unwrap_err(),
        AtlasLinalgError::InvalidInputRank {
            op: "batched_diag",
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
