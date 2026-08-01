use atlas_ndarray::{NDArray, Numeric};
use num_traits::Float;

use crate::core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand};

use super::{copy_matrix_row_major, is_symmetric, tolerance, validate_rank_two, zero_matrix_data};

#[derive(Clone, Debug)]
pub struct CholeskyFactorization<T: Numeric> {
    pub l: NDArray<T>,
}

pub fn cholesky<'a, T, M>(matrix: M) -> AtlasLinalgResult<CholeskyFactorization<T>>
where
    T: Numeric + Float + 'a,
    M: Into<LinalgOperand<'a, T>>,
{
    let matrix = matrix.into();
    let (rows, cols) = validate_rank_two(&matrix, "cholesky")?;

    if rows != cols {
        return Err(AtlasLinalgError::InvalidInputShape {
            op: "cholesky",
            shape: matrix.shape().to_vec(),
            reason: "Cholesky requires a square matrix",
        });
    }

    let n = rows;
    let tolerance = tolerance::<T>();
    let a = copy_matrix_row_major(&matrix);

    if !is_symmetric(&a, n, tolerance) {
        return Err(AtlasLinalgError::InvalidInputShape {
            op: "cholesky",
            shape: matrix.shape().to_vec(),
            reason: "Cholesky requires a symmetric matrix",
        });
    }

    let mut l = zero_matrix_data(n, n);

    for row in 0..n {
        for col in 0..=row {
            let mut value = a[row * n + col];

            for inner in 0..col {
                value -= l[row * n + inner] * l[col * n + inner];
            }

            if row == col {
                if value <= tolerance {
                    return Err(AtlasLinalgError::NotPositiveDefinite {
                        op: "cholesky",
                        index: row,
                    });
                }

                l[row * n + col] = value.sqrt();
            } else {
                l[row * n + col] = value / l[col * n + col];
            }
        }
    }

    Ok(CholeskyFactorization { l: NDArray::from_shape_vec([n, n], l)? })
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use crate::{AtlasLinalgError, cholesky, matmul};

    fn assert_close_slice(actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() <= tolerance);
        }
    }

    #[test]
    fn cholesky_returns_lower_triangular_factor() {
        let matrix = NDArray::from_shape_vec([2, 2], vec![4.0_f64, 2.0, 2.0, 3.0]).unwrap();
        let factor = cholesky(&matrix).unwrap();

        assert_eq!(factor.l.shape(), &[2, 2]);

        let reconstructed = matmul(&factor.l, factor.l.view().transpose()).unwrap();
        assert_close_slice(reconstructed.data(), matrix.data(), 1e-10);
    }

    #[test]
    fn cholesky_reports_expected_validation_errors() {
        let non_symmetric = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        let non_spd = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 2.0, 2.0, 1.0]).unwrap();

        assert!(matches!(
            cholesky(&non_symmetric).unwrap_err(),
            AtlasLinalgError::InvalidInputShape { op: "cholesky", .. }
        ));
        assert!(matches!(
            cholesky(&non_spd).unwrap_err(),
            AtlasLinalgError::NotPositiveDefinite { op: "cholesky", .. }
        ));
    }
}
