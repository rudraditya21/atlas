use atlas_ndarray::{NDArray, Numeric};
use num_traits::Float;

use crate::{
    core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand},
    internal::factorization::{
        copy_matrix_row_major, dot_slice, validate_finite, validate_rank_two,
        validate_upper_triangular, vector_norm, zero_matrix_data,
    },
};

#[derive(Clone, Debug)]
pub struct QrFactorization<T: Numeric> {
    q: NDArray<T>,
    r: NDArray<T>,
}

impl<T: Numeric> QrFactorization<T> {
    /// Returns the reduced orthonormal factor.
    pub fn q(&self) -> &NDArray<T> {
        &self.q
    }

    /// Returns the upper-triangular factor.
    pub fn r(&self) -> &NDArray<T> {
        &self.r
    }
}

impl<T: Numeric + Float> QrFactorization<T> {
    pub fn least_squares<'a, R>(&self, rhs: R) -> AtlasLinalgResult<NDArray<T>>
    where
        T: 'a,
        R: Into<LinalgOperand<'a, T>>,
    {
        self.solve_r(&self.apply_q_transpose(rhs)?)
    }

    pub fn apply_q_transpose<'a, R>(&self, rhs: R) -> AtlasLinalgResult<NDArray<T>>
    where
        T: 'a,
        R: Into<LinalgOperand<'a, T>>,
    {
        let (rows, columns) = self.q_shape("apply_q_transpose")?;
        let rhs = rhs.into();
        let (rhs_columns, vector_rhs) = match rhs.shape() {
            [rhs_rows] if *rhs_rows == rows => (1, true),
            [rhs_rows, rhs_columns] if *rhs_rows == rows => (*rhs_columns, false),
            [..] if rhs.ndim() == 1 || rhs.ndim() == 2 => {
                return Err(AtlasLinalgError::ShapeMismatch {
                    op: "apply_q_transpose",
                    left: vec![rows, columns],
                    right: rhs.shape().to_vec(),
                    reason: "right-hand side row count must match Q row count",
                });
            }
            _ => {
                return Err(AtlasLinalgError::InvalidInputRank {
                    op: "apply_q_transpose",
                    expected: "a vector or matrix",
                    rank: rhs.ndim(),
                });
            }
        };
        let mut values = vec![T::zero(); columns * rhs_columns];

        for column in 0..columns {
            for rhs_column in 0..rhs_columns {
                let mut value = T::zero();
                for row in 0..rows {
                    value +=
                        self.q.data()[row * columns + column] * rhs_value(&rhs, row, rhs_column);
                }
                values[column * rhs_columns + rhs_column] = value;
            }
        }

        if vector_rhs {
            NDArray::from_shape_vec([columns], values).map_err(Into::into)
        } else {
            NDArray::from_shape_vec([columns, rhs_columns], values).map_err(Into::into)
        }
    }

    pub fn solve_r<'a, R>(&self, rhs: R) -> AtlasLinalgResult<NDArray<T>>
    where
        T: 'a,
        R: Into<LinalgOperand<'a, T>>,
    {
        let (_, order) = self.q_shape("solve_r")?;
        let rhs = rhs.into();
        let (rhs_columns, vector_rhs) = match rhs.shape() {
            [rhs_rows] if *rhs_rows == order => (1, true),
            [rhs_rows, rhs_columns] if *rhs_rows == order => (*rhs_columns, false),
            [..] if rhs.ndim() == 1 || rhs.ndim() == 2 => {
                return Err(AtlasLinalgError::ShapeMismatch {
                    op: "solve_r",
                    left: vec![order, order],
                    right: rhs.shape().to_vec(),
                    reason: "right-hand side row count must match R row count",
                });
            }
            _ => {
                return Err(AtlasLinalgError::InvalidInputRank {
                    op: "solve_r",
                    expected: "a vector or matrix",
                    rank: rhs.ndim(),
                });
            }
        };
        let mut values = vec![T::zero(); order * rhs_columns];

        for row in (0..order).rev() {
            let diagonal = self.r.data()[row * order + row];
            if diagonal.is_zero() {
                return Err(AtlasLinalgError::SingularMatrix { op: "solve_r", pivot: row });
            }

            for column in 0..rhs_columns {
                let mut value = rhs_value(&rhs, row, column);
                for next_row in (row + 1)..order {
                    value -= self.r.data()[row * order + next_row]
                        * values[next_row * rhs_columns + column];
                }
                values[row * rhs_columns + column] = value / diagonal;
            }
        }

        if vector_rhs {
            NDArray::from_shape_vec([order], values).map_err(Into::into)
        } else {
            NDArray::from_shape_vec([order, rhs_columns], values).map_err(Into::into)
        }
    }

    fn q_shape(&self, op: &'static str) -> AtlasLinalgResult<(usize, usize)> {
        let [rows, columns] = self.q.shape() else {
            return Err(AtlasLinalgError::InvalidInputShape {
                op,
                shape: self.q.shape().to_vec(),
                reason: "QR Q factor must be a matrix",
            });
        };

        if self.r.shape() != [*columns, *columns] {
            return Err(AtlasLinalgError::InvalidInputShape {
                op,
                shape: self.r.shape().to_vec(),
                reason: "QR R factor must be square with Q column count",
            });
        }

        validate_upper_triangular(&self.r, op, "QR upper")?;

        Ok((*rows, *columns))
    }
}

fn rhs_value<T: Numeric>(rhs: &LinalgOperand<'_, T>, row: usize, column: usize) -> T {
    let offset = rhs.offset() + row * rhs.strides()[0];

    if rhs.ndim() == 1 {
        rhs.data()[offset]
    } else {
        rhs.data()[offset + column * rhs.strides()[1]]
    }
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
    validate_finite(&a, "qr")?;
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

pub fn least_squares<'a, 'b, T, M, R>(matrix: M, rhs: R) -> AtlasLinalgResult<NDArray<T>>
where
    T: Numeric + Float + 'a + 'b,
    M: Into<LinalgOperand<'a, T>>,
    R: Into<LinalgOperand<'b, T>>,
{
    qr(matrix)?.least_squares(rhs)
}

/// Returns the numerical rank using the QR tolerance `||a_j|| * ε * max(m, n)`.
pub fn matrix_rank<'a, T, M>(matrix: M) -> AtlasLinalgResult<usize>
where
    T: Numeric + Float + 'a,
    M: Into<LinalgOperand<'a, T>>,
{
    let matrix = matrix.into();
    let (rows, columns) = validate_rank_two(&matrix, "matrix_rank")?;
    let matrix = copy_matrix_row_major(&matrix);
    let mut basis = vec![T::zero(); rows * rows.min(columns)];
    let mut work = vec![T::zero(); rows];
    let mut rank = 0;

    for column in 0..columns {
        copy_column_from_row_major(&matrix, rows, columns, column, &mut work);
        let column_norm = vector_norm(&work);

        for prior in 0..rank {
            let basis_column = &basis[prior * rows..(prior + 1) * rows];
            let projection = dot_slice(basis_column, &work);
            subtract_projection(&mut work, basis_column, projection);
        }
        for prior in 0..rank {
            let basis_column = &basis[prior * rows..(prior + 1) * rows];
            let correction = dot_slice(basis_column, &work);
            subtract_projection(&mut work, basis_column, correction);
        }

        let residual_norm = vector_norm(&work);
        if residual_norm <= qr_rank_tolerance(column_norm, rows, columns) {
            continue;
        }

        let basis_column = &mut basis[rank * rows..(rank + 1) * rows];
        for (slot, &value) in basis_column.iter_mut().zip(work.iter()) {
            *slot = value / residual_norm;
        }
        rank += 1;
    }

    Ok(rank)
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

        assert_eq!(factors.q().shape(), &[3, 2]);
        assert_eq!(factors.r().shape(), &[2, 2]);

        let reconstructed = matmul(factors.q(), factors.r()).unwrap();
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
                1.0,
                1.0,
                1.0,
                1.0,
                1.0 + epsilon,
                1.0,
                1.0,
                1.0,
                1.0 + epsilon,
                1.0,
                1.0 + epsilon,
                1.0 + epsilon,
            ],
        )
        .unwrap();
        let factors = qr(&matrix).unwrap();

        let gram = matmul(factors.q().view().transpose(), factors.q()).unwrap();
        assert_close_slice(gram.data(), &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 1.0e-6);
    }
}
