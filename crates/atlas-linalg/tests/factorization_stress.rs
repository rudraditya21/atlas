use atlas_linalg::{cholesky, lu, matmul, qr};
use atlas_ndarray::NDArray;

fn assert_close_relative(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1e-8 * expected.abs().max(1.0));
    }
}

#[test]
fn factorization_reconstruction_handles_deterministic_stress_matrices() {
    let cases = [
        NDArray::from_shape_vec(
            [3, 3],
            vec![1.0_f64, 0.5, 1.0 / 3.0, 0.5, 1.0 / 3.0, 0.25, 1.0 / 3.0, 0.25, 0.2],
        )
        .unwrap(),
        NDArray::from_shape_vec([3, 3], vec![1.0e8_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0e-2])
            .unwrap(),
        NDArray::from_shape_vec([3, 3], vec![5.0_f64, 1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 5.0])
            .unwrap(),
        NDArray::from_shape_vec([3, 3], vec![1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0e-7])
            .unwrap(),
    ];

    for matrix in cases {
        let lu_factors = lu(&matrix).unwrap();
        let qr_factors = qr(&matrix).unwrap();
        let cholesky_factor = cholesky(&matrix).unwrap();

        let permuted = matmul(lu_factors.p(), &matrix).unwrap();
        let lu_reconstructed = matmul(lu_factors.l(), lu_factors.u()).unwrap();
        let qr_reconstructed = matmul(qr_factors.q(), qr_factors.r()).unwrap();
        let cholesky_reconstructed =
            matmul(cholesky_factor.l(), cholesky_factor.l().view().transpose()).unwrap();

        assert_close_relative(lu_reconstructed.data(), permuted.data());
        assert_close_relative(qr_reconstructed.data(), matrix.data());
        assert_close_relative(cholesky_reconstructed.data(), matrix.data());
    }
}
