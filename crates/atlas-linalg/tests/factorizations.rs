use atlas_linalg::{
    AtlasLinalgError, AtlasLinalgResult, CholeskyFactorization, LUFactorization, QRFactorization,
    cholesky, lu, matmul, qr,
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

    assert_shape(&factors.p, &[3, 3]);
    assert_shape(&factors.l, &[3, 3]);
    assert_shape(&factors.u, &[3, 3]);

    let permuted = matmul(&factors.p, &matrix).unwrap();
    let reconstructed = matmul(&factors.l, &factors.u).unwrap();

    assert_shape(&permuted, &[3, 3]);
    assert_shape(&reconstructed, &[3, 3]);
    assert_close_slice(permuted.data(), reconstructed.data(), 1e-10);
}

#[test]
fn qr_reconstructs_tall_input_and_has_reduced_shapes() {
    let matrix = NDArray::from_shape_vec([3, 2], vec![1.0_f64, 1.0, 1.0, 0.0, 0.0, 1.0]).unwrap();

    let factors: QRFactorization<f64> = expect_ok(qr(&matrix));

    assert_shape(&factors.q, &[3, 2]);
    assert_shape(&factors.r, &[2, 2]);

    let reconstructed = matmul(&factors.q, &factors.r).unwrap();
    assert_shape(&reconstructed, &[3, 2]);
    assert_close_slice(reconstructed.data(), matrix.data(), 1e-10);

    let gram = matmul(factors.q.view().transpose(), &factors.q).unwrap();
    assert_shape(&gram, &[2, 2]);
    assert_close_slice(gram.data(), &[1.0, 0.0, 0.0, 1.0], 1e-10);
}

#[test]
fn cholesky_reconstructs_symmetric_positive_definite_input() {
    let matrix = NDArray::from_shape_vec([2, 2], vec![4.0_f64, 2.0, 2.0, 3.0]).unwrap();

    let factor: CholeskyFactorization<f64> = expect_ok(cholesky(&matrix));

    assert_shape(&factor.l, &[2, 2]);

    let reconstructed = matmul(&factor.l, factor.l.view().transpose()).unwrap();
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
    let permuted = matmul(&factors.p, &matrix).unwrap();
    let reconstructed = matmul(&factors.l, &factors.u).unwrap();

    assert_shape(&factors.p, &[4, 4]);
    assert_shape(&factors.l, &[4, 4]);
    assert_shape(&factors.u, &[4, 4]);
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
    let reconstructed = matmul(&factors.q, &factors.r).unwrap();
    let gram = matmul(factors.q.view().transpose(), &factors.q).unwrap();

    assert_shape(&factors.q, &[3, 3]);
    assert_shape(&factors.r, &[3, 3]);
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
    let reconstructed = matmul(&factor.l, factor.l.view().transpose()).unwrap();

    assert_shape(&factor.l, &[3, 3]);
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
