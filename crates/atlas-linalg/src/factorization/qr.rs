use atlas_ndarray::{NDArray, Numeric};
use num_traits::Float;

use crate::core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand};

use super::{
    column_from_storage, copy_matrix_row_major, dot_slice, extract_column, tolerance,
    validate_rank_two, vector_norm, zero_matrix_data,
};

#[derive(Clone, Debug)]
pub struct QrFactorization<T: Numeric> {
    pub q: NDArray<T>,
    pub r: NDArray<T>,
}

pub fn qr<'a, T, M>(matrix: M) -> AtlasLinalgResult<QrFactorization<T>>
where
    T: Numeric + Float + 'a,
    M: Into<LinalgOperand<'a, T>>,
{
    let matrix = matrix.into();
    let (rows, cols) = validate_rank_two(&matrix, "qr")?;

    if rows < cols {
        return Err(AtlasLinalgError::InvalidInputShape {
            op: "qr",
            shape: matrix.shape().to_vec(),
            reason: "QR currently requires rows >= columns",
        });
    }

    let tolerance = tolerance::<T>();
    let a = copy_matrix_row_major(&matrix);
    let mut q_columns = vec![T::zero(); rows * cols];
    let mut r = zero_matrix_data(cols, cols);

    for col in 0..cols {
        let mut v = extract_column(&a, rows, cols, col);

        for prior in 0..col {
            let q_col = column_from_storage(&q_columns, rows, cols, prior);
            let projection = dot_slice(&q_col, &v);
            r[prior * cols + col] = projection;

            for row in 0..rows {
                v[row] -= projection * q_columns[row * cols + prior];
            }
        }

        let norm = vector_norm(&v);

        if norm <= tolerance {
            return Err(AtlasLinalgError::RankDeficientMatrix { op: "qr", column: col });
        }

        r[col * cols + col] = norm;

        for row in 0..rows {
            q_columns[row * cols + col] = v[row] / norm;
        }
    }

    Ok(QrFactorization {
        q: NDArray::from_shape_vec([rows, cols], q_columns)?,
        r: NDArray::from_shape_vec([cols, cols], r)?,
    })
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use crate::{AtlasLinalgError, matmul, qr};

    fn assert_close_slice(actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() <= tolerance);
        }
    }

    #[test]
    fn qr_returns_reduced_factors() {
        let matrix =
            NDArray::from_shape_vec([3, 2], vec![1.0_f64, 1.0, 1.0, 0.0, 0.0, 1.0]).unwrap();
        let factors = qr(&matrix).unwrap();

        assert_eq!(factors.q.shape(), &[3, 2]);
        assert_eq!(factors.r.shape(), &[2, 2]);

        let reconstructed = matmul(&factors.q, &factors.r).unwrap();
        assert_close_slice(reconstructed.data(), matrix.data(), 1e-10);
    }

    #[test]
    fn qr_reports_expected_validation_errors() {
        let wide = NDArray::from_shape_vec([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let rank_deficient = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 2.0, 2.0, 4.0]).unwrap();

        assert!(matches!(
            qr(&wide).unwrap_err(),
            AtlasLinalgError::InvalidInputShape { op: "qr", .. }
        ));
        assert!(matches!(
            qr(&rank_deficient).unwrap_err(),
            AtlasLinalgError::RankDeficientMatrix { op: "qr", .. }
        ));
    }
}
