use atlas_linalg::{AtlasLinalgError, cholesky, lu, matmul, qr};
use atlas_ndarray::NDArray;

fn assert_close_slice(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() <= tolerance);
    }
}

#[test]
fn lu_reconstructs_permuted_input() {
    let matrix = NDArray::from_shape_vec(
        [3, 3],
        vec![0.0_f64, 2.0, 1.0, 1.0, 1.0, 0.0, 2.0, 1.0, 1.0],
    )
    .unwrap();

    let factors = lu(&matrix).unwrap();

    assert_eq!(factors.p.shape(), &[3, 3]);
    assert_eq!(factors.l.shape(), &[3, 3]);
    assert_eq!(factors.u.shape(), &[3, 3]);

    let permuted = matmul(&factors.p, &matrix).unwrap();
    let reconstructed = matmul(&factors.l, &factors.u).unwrap();

    assert_close_slice(permuted.data(), reconstructed.data(), 1e-10);
}

#[test]
fn qr_reconstructs_tall_input_and_has_reduced_shapes() {
    let matrix = NDArray::from_shape_vec([3, 2], vec![1.0_f64, 1.0, 1.0, 0.0, 0.0, 1.0]).unwrap();

    let factors = qr(&matrix).unwrap();

    assert_eq!(factors.q.shape(), &[3, 2]);
    assert_eq!(factors.r.shape(), &[2, 2]);

    let reconstructed = matmul(&factors.q, &factors.r).unwrap();
    assert_close_slice(reconstructed.data(), matrix.data(), 1e-10);

    let gram = matmul(factors.q.view().transpose(), &factors.q).unwrap();
    assert_close_slice(gram.data(), &[1.0, 0.0, 0.0, 1.0], 1e-10);
}

#[test]
fn cholesky_reconstructs_symmetric_positive_definite_input() {
    let matrix = NDArray::from_shape_vec([2, 2], vec![4.0_f64, 2.0, 2.0, 3.0]).unwrap();

    let factor = cholesky(&matrix).unwrap();

    assert_eq!(factor.l.shape(), &[2, 2]);

    let reconstructed = matmul(&factor.l, factor.l.view().transpose()).unwrap();
    assert_close_slice(reconstructed.data(), matrix.data(), 1e-10);
}

#[test]
fn factorization_failures_are_structured() {
    let rank_deficient =
        NDArray::from_shape_vec([3, 2], vec![1.0_f64, 2.0, 2.0, 4.0, 3.0, 6.0]).unwrap();
    let non_symmetric = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 2.0, 0.0, 1.0]).unwrap();

    assert!(matches!(
        qr(&rank_deficient).unwrap_err(),
        AtlasLinalgError::RankDeficientMatrix { op: "qr", .. }
    ));
    assert!(matches!(
        cholesky(&non_symmetric).unwrap_err(),
        AtlasLinalgError::InvalidInputShape { op: "cholesky", .. }
    ));
}
