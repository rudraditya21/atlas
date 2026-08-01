use atlas_ndarray::{NDArray, Numeric};
use num_traits::Float;

use crate::core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand};
use crate::internal::factorization::{
    copy_matrix_row_major, dot_slice, tolerance, validate_rank_two, vector_norm, zero_matrix_data,
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
    let mut v = vec![T::zero(); rows];

    for col in 0..cols {
        for row in 0..rows {
            v[row] = a[row * cols + col];
        }

        for prior in 0..col {
            let q_col = &q_columns[prior * rows..(prior + 1) * rows];
            let projection = dot_slice(q_col, &v);
            r[prior * cols + col] = projection;

            for (value, &basis) in v.iter_mut().zip(q_col.iter()) {
                *value -= projection * basis;
            }
        }

        let norm = vector_norm(&v);

        if norm <= tolerance {
            return Err(AtlasLinalgError::RankDeficientMatrix { op: "qr", column: col });
        }

        r[col * cols + col] = norm;
        let q_col = &mut q_columns[col * rows..(col + 1) * rows];

        for (slot, &value) in q_col.iter_mut().zip(v.iter()) {
            *slot = value / norm;
        }
    }

    let q = column_major_to_row_major(&q_columns, rows, cols);

    Ok(QrFactorization {
        q: NDArray::from_shape_vec([rows, cols], q)?,
        r: NDArray::from_shape_vec([cols, cols], r)?,
    })
}

fn column_major_to_row_major<T: Numeric>(data: &[T], rows: usize, cols: usize) -> Vec<T> {
    let mut reordered = vec![T::zero(); rows * cols];

    for col in 0..cols {
        let source = &data[col * rows..(col + 1) * rows];

        for row in 0..rows {
            reordered[row * cols + col] = source[row];
        }
    }

    reordered
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
