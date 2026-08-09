use atlas_ndarray::{NDArray, Numeric};
use num_traits::Float;

use crate::core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand};
use crate::internal::factorization::{
    copy_matrix_row_major, dot_slice, validate_rank_two, vector_norm, zero_matrix_data,
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

    let a = copy_matrix_row_major(&matrix);
    let mut q_columns = vec![T::zero(); rows * cols];
    let mut r = zero_matrix_data(cols, cols);
    let mut work = vec![T::zero(); rows];

    for col in 0..cols {
        copy_column_from_row_major(&a, rows, cols, col, &mut work);
        let column_norm = vector_norm(&work);

        for prior in 0..col {
            let q_col = &q_columns[prior * rows..(prior + 1) * rows];
            let projection = dot_slice(q_col, &work);
            r[prior * cols + col] = projection;
            subtract_projection(&mut work, q_col, projection);
        }

        // Reorthogonalize once to recover orthogonality lost to rounding
        // when columns are nearly linearly dependent.
        for prior in 0..col {
            let q_col = &q_columns[prior * rows..(prior + 1) * rows];
            let correction = dot_slice(q_col, &work);
            r[prior * cols + col] += correction;
            subtract_projection(&mut work, q_col, correction);
        }

        let norm = vector_norm(&work);

        if norm <= qr_rank_tolerance(column_norm, rows, cols) {
            return Err(AtlasLinalgError::RankDeficientMatrix { op: "qr", column: col });
        }

        r[col * cols + col] = norm;
        let q_col = &mut q_columns[col * rows..(col + 1) * rows];

        for (slot, &value) in q_col.iter_mut().zip(work.iter()) {
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

fn copy_column_from_row_major<T: Numeric>(
    matrix: &[T],
    rows: usize,
    cols: usize,
    col: usize,
    out: &mut [T],
) {
    for row in 0..rows {
        out[row] = matrix[row * cols + col];
    }
}

fn subtract_projection<T: Numeric>(vector: &mut [T], basis: &[T], scale: T) {
    for (value, &basis_value) in vector.iter_mut().zip(basis.iter()) {
        *value -= scale * basis_value;
    }
}

fn qr_rank_tolerance<T: Float>(column_norm: T, rows: usize, cols: usize) -> T {
    let dimension_scale = T::from(rows.max(cols)).unwrap_or(T::one());
    column_norm * T::epsilon() * dimension_scale
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

    #[test]
    fn qr_preserves_orthogonality_for_nearly_dependent_columns() {
        let epsilon = 1.0e-10_f64;
        let matrix = NDArray::from_shape_vec(
            [4, 3],
            vec![
                1.0, 1.0, 1.0,
                1.0, 1.0 + epsilon, 1.0,
                1.0, 1.0, 1.0 + epsilon,
                1.0, 1.0 + epsilon, 1.0 + epsilon,
            ],
        )
        .unwrap();
        let factors = qr(&matrix).unwrap();

        let gram = matmul(factors.q.view().transpose(), &factors.q).unwrap();
        assert_close_slice(
            gram.data(),
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            1.0e-6,
        );
    }
}
