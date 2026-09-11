use atlas_linalg::{
    AtlasLinalgError, AtlasLinalgResult, CholeskyFactorization, LUFactorization, QRFactorization,
    cholesky, det, least_squares, lu, matmul, qr, slogdet, solve, solve_lower_triangular,
    solve_spd, solve_upper_triangular,
};
use atlas_ndarray::NDArray;

fn assert_close_slice(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() <= tolerance);
    }
}

fn assert_shape<T>(array: &NDArray<T>, expected: &[usize])
where
    T: atlas_ndarray::Numeric,
{
    assert_eq!(array.shape(), expected);
}

fn expect_ok<T>(result: AtlasLinalgResult<T>) -> T {
    result.unwrap()
}

#[test]
fn lu_reconstructs_permuted_input() {
    let matrix =
        NDArray::from_shape_vec([3, 3], vec![0.0_f64, 2.0, 1.0, 1.0, 1.0, 0.0, 2.0, 1.0, 1.0])
            .unwrap();

    let factors: LUFactorization<f64> = expect_ok(lu(&matrix));

    assert_shape(factors.p(), &[3, 3]);
    assert_shape(factors.l(), &[3, 3]);
    assert_shape(factors.u(), &[3, 3]);

    let permuted = matmul(factors.p(), &matrix).unwrap();
    let reconstructed = matmul(factors.l(), factors.u()).unwrap();

    assert_shape(&permuted, &[3, 3]);
    assert_shape(&reconstructed, &[3, 3]);
    assert_close_slice(permuted.data(), reconstructed.data(), 1e-10);
}

#[test]
fn lu_determinants_preserve_sign_and_permutation_parity() {
    let positive = NDArray::from_shape_vec([2, 2], vec![2.0_f64, 0.0, 0.0, 3.0]).unwrap();
    let negative = NDArray::from_shape_vec([2, 2], vec![2.0_f64, 0.0, 0.0, -3.0]).unwrap();
    let pivoted = NDArray::from_shape_vec([2, 2], vec![0.0_f64, 2.0, 1.0, 3.0]).unwrap();

    assert_eq!(det(&positive).unwrap(), 6.0);
    assert_eq!(lu(&negative).unwrap().det().unwrap(), -6.0);
    assert_eq!(lu(&pivoted).unwrap().det().unwrap(), -2.0);
}

#[test]
fn slogdet_reconstructs_determinants_and_accepts_views() {
    let negative = NDArray::from_shape_vec([2, 2], vec![2.0_f64, 0.0, 0.0, -3.0]).unwrap();
    let (sign, log_abs_det) = slogdet(&negative).unwrap();
    let base = NDArray::from_shape_vec(
        [3, 4],
        vec![9.0_f64, 2.0, 0.0, 0.0, 8.0, 0.0, 3.0, 0.0, 7.0, 0.0, 0.0, 4.0],
    )
    .unwrap();
    let view = base.view().slice([0, 1], [3, 3]).unwrap();
    let (view_sign, view_log_abs_det) = slogdet(view).unwrap();

    assert_eq!(sign, -1.0);
    assert!((sign * log_abs_det.exp() + 6.0).abs() <= 1e-10);
    assert_eq!(view_sign, 1.0);
    assert!((view_sign * view_log_abs_det.exp() - 24.0).abs() <= 1e-10);
}

#[test]
fn determinant_apis_reject_singular_input() {
    let singular = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 2.0, 2.0, 4.0]).unwrap();

    assert!(matches!(det(&singular), Err(AtlasLinalgError::SingularMatrix { op: "lu", .. })));
    assert!(matches!(slogdet(&singular), Err(AtlasLinalgError::SingularMatrix { op: "lu", .. })));
}

#[test]
fn triangular_solvers_support_vector_and_matrix_right_hand_sides() {
    let lower = NDArray::from_shape_vec([2, 2], vec![2.0_f64, 0.0, 3.0, 1.0]).unwrap();
    let upper = NDArray::from_shape_vec([2, 2], vec![2.0_f64, 3.0, 0.0, 1.0]).unwrap();
    let lower_rhs = NDArray::from_shape_vec([2], vec![4.0_f64, 5.0]).unwrap();
    let upper_rhs = NDArray::from_shape_vec([2], vec![1.0_f64, -1.0]).unwrap();
    let lower_multiple_rhs = NDArray::from_shape_vec([2, 2], vec![4.0_f64, 2.0, 5.0, 6.0]).unwrap();
    let upper_multiple_rhs =
        NDArray::from_shape_vec([2, 2], vec![1.0_f64, 11.0, -1.0, 3.0]).unwrap();

    assert_close_slice(
        solve_lower_triangular(&lower, &lower_rhs).unwrap().data(),
        &[2.0, -1.0],
        1e-10,
    );
    assert_close_slice(
        solve_lower_triangular(&lower, &lower_multiple_rhs).unwrap().data(),
        &[2.0, 1.0, -1.0, 3.0],
        1e-10,
    );
    assert_close_slice(
        solve_upper_triangular(&upper, &upper_rhs).unwrap().data(),
        &[2.0, -1.0],
        1e-10,
    );
    assert_close_slice(
        solve_upper_triangular(&upper, &upper_multiple_rhs).unwrap().data(),
        &[2.0, 1.0, -1.0, 3.0],
        1e-10,
    );
}

#[test]
fn triangular_solvers_report_validation_and_singular_errors() {
    let vector = NDArray::from_shape_vec([2], vec![1.0_f64, 2.0]).unwrap();
    let non_square =
        NDArray::from_shape_vec([2, 3], vec![1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
    let lower = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 0.0, 1.0, 1.0]).unwrap();
    let mismatched_rhs = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let scalar_rhs = NDArray::from_shape_vec([], vec![1.0_f64]).unwrap();
    let singular_lower = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 0.0, 1.0, 0.0]).unwrap();
    let singular_upper = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 1.0, 0.0, 0.0]).unwrap();

    assert!(matches!(
        solve_lower_triangular(&vector, &vector),
        Err(AtlasLinalgError::InvalidInputRank { op: "solve_lower_triangular", rank: 1, .. })
    ));
    assert!(matches!(
        solve_upper_triangular(&non_square, &vector),
        Err(AtlasLinalgError::InvalidInputShape { op: "solve_upper_triangular", .. })
    ));
    assert!(matches!(
        solve_lower_triangular(&lower, &mismatched_rhs),
        Err(AtlasLinalgError::ShapeMismatch { op: "solve_lower_triangular", .. })
    ));
    assert!(matches!(
        solve_upper_triangular(&lower, &scalar_rhs),
        Err(AtlasLinalgError::InvalidInputRank { op: "solve_upper_triangular", rank: 0, .. })
    ));
    assert!(matches!(
        solve_lower_triangular(&singular_lower, &vector),
        Err(AtlasLinalgError::SingularMatrix { op: "solve_lower_triangular", pivot: 1 })
    ));
    assert!(matches!(
        solve_upper_triangular(&singular_upper, &vector),
        Err(AtlasLinalgError::SingularMatrix { op: "solve_upper_triangular", pivot: 1 })
    ));
}

#[test]
fn qr_reconstructs_tall_input_and_has_reduced_shapes() {
    let matrix = NDArray::from_shape_vec([3, 2], vec![1.0_f64, 1.0, 1.0, 0.0, 0.0, 1.0]).unwrap();

    let factors: QRFactorization<f64> = expect_ok(qr(&matrix));

    assert_shape(factors.q(), &[3, 2]);
    assert_shape(factors.r(), &[2, 2]);

    let reconstructed = matmul(factors.q(), factors.r()).unwrap();
    assert_shape(&reconstructed, &[3, 2]);
    assert_close_slice(reconstructed.data(), matrix.data(), 1e-10);

    let gram = matmul(factors.q().view().transpose(), factors.q()).unwrap();
    assert_shape(&gram, &[2, 2]);
    assert_close_slice(gram.data(), &[1.0, 0.0, 0.0, 1.0], 1e-10);
}

#[test]
fn cholesky_reconstructs_symmetric_positive_definite_input() {
    let matrix = NDArray::from_shape_vec([2, 2], vec![4.0_f64, 2.0, 2.0, 3.0]).unwrap();

    let factor: CholeskyFactorization<f64> = expect_ok(cholesky(&matrix));

    assert_shape(factor.l(), &[2, 2]);

    let reconstructed = matmul(factor.l(), factor.l().view().transpose()).unwrap();
    assert_shape(&reconstructed, &[2, 2]);
    assert_close_slice(reconstructed.data(), matrix.data(), 1e-10);
}

#[test]
fn lu_reconstructs_four_by_four_input_and_preserves_square_shapes() {
    let matrix = NDArray::from_shape_vec(
        [4, 4],
        vec![0.0_f64, 2.0, 1.0, 3.0, 4.0, 5.0, 2.0, 1.0, 6.0, 7.0, 8.0, 2.0, 3.0, 1.0, 5.0, 9.0],
    )
    .unwrap();

    let factors = lu(&matrix).unwrap();
    let permuted = matmul(factors.p(), &matrix).unwrap();
    let reconstructed = matmul(factors.l(), factors.u()).unwrap();

    assert_shape(factors.p(), &[4, 4]);
    assert_shape(factors.l(), &[4, 4]);
    assert_shape(factors.u(), &[4, 4]);
    assert_shape(&permuted, &[4, 4]);
    assert_shape(&reconstructed, &[4, 4]);
    assert_close_slice(permuted.data(), reconstructed.data(), 1e-10);
}

#[test]
fn qr_reconstructs_square_input_with_square_q_and_r_shapes() {
    let matrix = NDArray::from_shape_vec(
        [3, 3],
        vec![12.0_f64, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0],
    )
    .unwrap();

    let factors = qr(&matrix).unwrap();
    let reconstructed = matmul(factors.q(), factors.r()).unwrap();
    let gram = matmul(factors.q().view().transpose(), factors.q()).unwrap();

    assert_shape(factors.q(), &[3, 3]);
    assert_shape(factors.r(), &[3, 3]);
    assert_shape(&reconstructed, &[3, 3]);
    assert_shape(&gram, &[3, 3]);
    assert_close_slice(reconstructed.data(), matrix.data(), 1e-10);
    assert_close_slice(gram.data(), &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 1e-10);
}

#[test]
fn cholesky_reconstructs_three_by_three_input_and_preserves_shape() {
    let matrix = NDArray::from_shape_vec(
        [3, 3],
        vec![25.0_f64, 15.0, -5.0, 15.0, 18.0, 0.0, -5.0, 0.0, 11.0],
    )
    .unwrap();

    let factor = cholesky(&matrix).unwrap();
    let reconstructed = matmul(factor.l(), factor.l().view().transpose()).unwrap();

    assert_shape(factor.l(), &[3, 3]);
    assert_shape(&reconstructed, &[3, 3]);
    assert_close_slice(reconstructed.data(), matrix.data(), 1e-10);
}

#[test]
fn lu_reports_exact_error_for_non_square_input() {
    let matrix = NDArray::from_shape_vec([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    assert_eq!(
        lu(&matrix).unwrap_err(),
        AtlasLinalgError::InvalidInputShape {
            op: "lu",
            shape: vec![2, 3],
            reason: "LU requires a square matrix",
        }
    );
}

#[test]
fn lu_reports_exact_error_for_singular_input() {
    let matrix = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 2.0, 2.0, 4.0]).unwrap();

    assert_eq!(lu(&matrix).unwrap_err(), AtlasLinalgError::SingularMatrix { op: "lu", pivot: 1 });
}

#[test]
fn solve_returns_vector_solution_and_preserves_the_residual() {
    let matrix = NDArray::from_shape_vec([2, 2], vec![3.0_f64, 1.0, 1.0, 2.0]).unwrap();
    let rhs = NDArray::from_shape_vec([2], vec![9.0_f64, 8.0]).unwrap();

    let solution = solve(&matrix, &rhs).unwrap();

    assert_shape(&solution, &[2]);
    assert_close_slice(solution.data(), &[2.0, 3.0], 1e-10);
    assert_close_slice(matmul(&matrix, &solution).unwrap().data(), rhs.data(), 1e-10);
}

#[test]
fn lu_factorization_solves_multiple_right_hand_sides() {
    let matrix = NDArray::from_shape_vec([2, 2], vec![3.0_f64, 1.0, 1.0, 2.0]).unwrap();
    let rhs = NDArray::from_shape_vec([2, 2], vec![9.0_f64, 5.0, 8.0, 5.0]).unwrap();

    let solution = lu(&matrix).unwrap().solve(&rhs).unwrap();

    assert_shape(&solution, &[2, 2]);
    assert_close_slice(solution.data(), &[2.0, 1.0, 3.0, 2.0], 1e-10);
    assert_close_slice(matmul(&matrix, &solution).unwrap().data(), rhs.data(), 1e-10);
}

#[test]
fn solve_applies_lu_pivoting_to_the_right_hand_side() {
    let matrix = NDArray::from_shape_vec([2, 2], vec![0.0_f64, 2.0, 1.0, 3.0]).unwrap();
    let rhs = NDArray::from_shape_vec([2], vec![4.0_f64, 7.0]).unwrap();

    let solution = solve(&matrix, &rhs).unwrap();

    assert_close_slice(solution.data(), &[1.0, 2.0], 1e-10);
    assert_close_slice(matmul(&matrix, &solution).unwrap().data(), rhs.data(), 1e-10);
}

#[test]
fn solve_reports_expected_validation_errors() {
    let non_square =
        NDArray::from_shape_vec([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let square = NDArray::from_shape_vec([2, 2], vec![3.0_f64, 1.0, 1.0, 2.0]).unwrap();
    let singular = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 2.0, 2.0, 4.0]).unwrap();
    let valid_rhs = NDArray::from_shape_vec([2], vec![1.0_f64, 2.0]).unwrap();
    let mismatched_rhs = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let scalar_rhs = NDArray::from_shape_vec([], vec![1.0_f64]).unwrap();

    assert_eq!(
        solve(&non_square, &valid_rhs).unwrap_err(),
        AtlasLinalgError::InvalidInputShape {
            op: "lu",
            shape: vec![2, 3],
            reason: "LU requires a square matrix",
        }
    );
    assert_eq!(
        solve(&singular, &valid_rhs).unwrap_err(),
        AtlasLinalgError::SingularMatrix { op: "lu", pivot: 1 }
    );
    assert_eq!(
        solve(&square, &mismatched_rhs).unwrap_err(),
        AtlasLinalgError::ShapeMismatch {
            op: "solve",
            left: vec![2, 2],
            right: vec![3],
            reason: "right-hand side row count must match coefficient matrix row count",
        }
    );
    assert_eq!(
        solve(&square, &scalar_rhs).unwrap_err(),
        AtlasLinalgError::InvalidInputRank { op: "solve", expected: "a vector or matrix", rank: 0 }
    );
}

#[test]
fn least_squares_solves_exact_and_overdetermined_systems() {
    let exact_matrix = NDArray::from_shape_vec([2, 2], vec![3.0_f64, 1.0, 1.0, 2.0]).unwrap();
    let exact_rhs = NDArray::from_shape_vec([2], vec![9.0_f64, 8.0]).unwrap();
    let overdetermined_matrix =
        NDArray::from_shape_vec([3, 2], vec![1.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let overdetermined_rhs = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 4.0]).unwrap();

    let exact_solution = least_squares(&exact_matrix, &exact_rhs).unwrap();
    let overdetermined_solution =
        least_squares(&overdetermined_matrix, &overdetermined_rhs).unwrap();
    let prediction = matmul(&overdetermined_matrix, &overdetermined_solution).unwrap();
    let residual = prediction
        .data()
        .iter()
        .zip(overdetermined_rhs.data())
        .map(|(predicted, observed)| predicted - observed)
        .collect::<Vec<_>>();

    assert_close_slice(exact_solution.data(), &[2.0, 3.0], 1e-10);
    assert_close_slice(overdetermined_solution.data(), &[4.0 / 3.0, 7.0 / 3.0], 1e-10);
    assert_close_slice(prediction.data(), &[4.0 / 3.0, 7.0 / 3.0, 11.0 / 3.0], 1e-10);
    assert_close_slice(&[residual[0] + residual[2], residual[1] + residual[2]], &[0.0, 0.0], 1e-10);
}

#[test]
fn qr_factorization_least_squares_supports_multiple_right_hand_sides() {
    let matrix = NDArray::from_shape_vec([3, 2], vec![1.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let rhs = NDArray::from_shape_vec([3, 2], vec![1.0_f64, 2.0, 2.0, 1.0, 3.0, 3.0]).unwrap();

    let solution = qr(&matrix).unwrap().least_squares(&rhs).unwrap();

    assert_shape(&solution, &[2, 2]);
    assert_close_slice(solution.data(), &[1.0, 2.0, 2.0, 1.0], 1e-10);
    assert_close_slice(matmul(&matrix, &solution).unwrap().data(), rhs.data(), 1e-10);
}

#[test]
fn least_squares_preserves_qr_validation_errors() {
    let wide = NDArray::from_shape_vec([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rank_deficient =
        NDArray::from_shape_vec([3, 2], vec![1.0_f64, 2.0, 2.0, 4.0, 3.0, 6.0]).unwrap();
    let rhs = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 3.0]).unwrap();

    assert_eq!(
        least_squares(&wide, &rhs).unwrap_err(),
        AtlasLinalgError::InvalidInputShape {
            op: "qr",
            shape: vec![2, 3],
            reason: "QR currently requires rows >= columns",
        }
    );
    assert_eq!(
        least_squares(&rank_deficient, &rhs).unwrap_err(),
        AtlasLinalgError::RankDeficientMatrix { op: "qr", column: 1 }
    );
}

#[test]
fn solve_spd_solves_exact_systems_and_multiple_right_hand_sides() {
    let matrix = NDArray::from_shape_vec([2, 2], vec![4.0_f64, 2.0, 2.0, 3.0]).unwrap();
    let rhs = NDArray::from_shape_vec([2], vec![8.0_f64, 8.0]).unwrap();
    let multiple_rhs = NDArray::from_shape_vec([2, 2], vec![8.0_f64, 10.0, 8.0, 7.0]).unwrap();

    let solution = solve_spd(&matrix, &rhs).unwrap();
    let multiple_solution = cholesky(&matrix).unwrap().solve(&multiple_rhs).unwrap();

    assert_close_slice(solution.data(), &[1.0, 2.0], 1e-10);
    assert_close_slice(matmul(&matrix, &solution).unwrap().data(), rhs.data(), 1e-10);
    assert_shape(&multiple_solution, &[2, 2]);
    assert_close_slice(multiple_solution.data(), &[1.0, 2.0, 2.0, 1.0], 1e-10);
    assert_close_slice(
        matmul(&matrix, &multiple_solution).unwrap().data(),
        multiple_rhs.data(),
        1e-10,
    );
}

#[test]
fn solve_spd_reports_expected_validation_errors() {
    let non_square =
        NDArray::from_shape_vec([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let non_symmetric = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 2.0, 0.0, 1.0]).unwrap();
    let non_spd = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 2.0, 2.0, 1.0]).unwrap();
    let matrix = NDArray::from_shape_vec([2, 2], vec![4.0_f64, 2.0, 2.0, 3.0]).unwrap();
    let rhs = NDArray::from_shape_vec([2], vec![1.0_f64, 2.0]).unwrap();
    let mismatched_rhs = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let scalar_rhs = NDArray::from_shape_vec([], vec![1.0_f64]).unwrap();

    assert_eq!(
        solve_spd(&non_square, &rhs).unwrap_err(),
        AtlasLinalgError::InvalidInputShape {
            op: "cholesky",
            shape: vec![2, 3],
            reason: "Cholesky requires a square matrix",
        }
    );
    assert_eq!(
        solve_spd(&non_symmetric, &rhs).unwrap_err(),
        AtlasLinalgError::InvalidInputShape {
            op: "cholesky",
            shape: vec![2, 2],
            reason: "Cholesky requires a symmetric matrix",
        }
    );
    assert_eq!(
        solve_spd(&non_spd, &rhs).unwrap_err(),
        AtlasLinalgError::NotPositiveDefinite { op: "cholesky", index: 1 }
    );
    assert_eq!(
        solve_spd(&matrix, &mismatched_rhs).unwrap_err(),
        AtlasLinalgError::ShapeMismatch {
            op: "solve_spd",
            left: vec![2, 2],
            right: vec![3],
            reason: "right-hand side row count must match coefficient matrix row count",
        }
    );
    assert_eq!(
        solve_spd(&matrix, &scalar_rhs).unwrap_err(),
        AtlasLinalgError::InvalidInputRank {
            op: "solve_spd",
            expected: "a vector or matrix",
            rank: 0,
        }
    );
}

#[test]
fn qr_reports_exact_error_for_wide_input() {
    let matrix = NDArray::from_shape_vec([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    assert_eq!(
        qr(&matrix).unwrap_err(),
        AtlasLinalgError::InvalidInputShape {
            op: "qr",
            shape: vec![2, 3],
            reason: "QR currently requires rows >= columns",
        }
    );
}

#[test]
fn qr_reports_exact_error_for_rank_deficient_input() {
    let matrix = NDArray::from_shape_vec([3, 2], vec![1.0_f64, 2.0, 2.0, 4.0, 3.0, 6.0]).unwrap();

    assert_eq!(
        qr(&matrix).unwrap_err(),
        AtlasLinalgError::RankDeficientMatrix { op: "qr", column: 1 }
    );
}

#[test]
fn cholesky_reports_exact_error_for_non_square_input() {
    let matrix = NDArray::from_shape_vec([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    assert_eq!(
        cholesky(&matrix).unwrap_err(),
        AtlasLinalgError::InvalidInputShape {
            op: "cholesky",
            shape: vec![2, 3],
            reason: "Cholesky requires a square matrix",
        }
    );
}

#[test]
fn cholesky_reports_exact_error_for_non_symmetric_input() {
    let matrix = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 2.0, 0.0, 1.0]).unwrap();

    assert_eq!(
        cholesky(&matrix).unwrap_err(),
        AtlasLinalgError::InvalidInputShape {
            op: "cholesky",
            shape: vec![2, 2],
            reason: "Cholesky requires a symmetric matrix",
        }
    );
}

#[test]
fn cholesky_reports_exact_error_for_non_positive_definite_input() {
    let matrix = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 2.0, 2.0, 1.0]).unwrap();

    assert_eq!(
        cholesky(&matrix).unwrap_err(),
        AtlasLinalgError::NotPositiveDefinite { op: "cholesky", index: 1 }
    );
}
