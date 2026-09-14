use atlas_ndarray::{NDArray, Numeric};
use num_traits::Float;

use crate::{
    core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand},
    dense::triangular::{solve_lower_triangular_with_op, solve_upper_triangular_with_op},
    internal::factorization::{
        copy_matrix_row_major, dot_slice, is_symmetric, tolerance, validate_finite,
        validate_lower_triangular, validate_rank_two, zero_matrix_data,
    },
};

#[derive(Clone, Debug)]
pub struct CholeskyFactorization<T: Numeric> {
    l: NDArray<T>,
}

impl<T: Numeric> CholeskyFactorization<T> {
    /// Returns the lower-triangular factor.
    pub fn l(&self) -> &NDArray<T> {
        &self.l
    }
}

impl<T: Numeric + Float> CholeskyFactorization<T> {
    pub fn solve<'a, R>(&self, rhs: R) -> AtlasLinalgResult<NDArray<T>>
    where
        T: 'a,
        R: Into<LinalgOperand<'a, T>>,
    {
        let rhs = rhs.into();
        self.solve_operand(&rhs)
    }

    fn solve_operand(&self, rhs: &LinalgOperand<'_, T>) -> AtlasLinalgResult<NDArray<T>> {
        let order = self.order()?;
        match rhs.shape() {
            [rows] if *rows == order => {}
            [rows, _] if *rows == order => {}
            [..] if rhs.ndim() == 1 || rhs.ndim() == 2 => {
                return Err(AtlasLinalgError::ShapeMismatch {
                    op: "solve_spd",
                    left: vec![order, order],
                    right: rhs.shape().to_vec(),
                    reason: "right-hand side row count must match coefficient matrix row count",
                });
            }
            _ => {
                return Err(AtlasLinalgError::InvalidInputRank {
                    op: "solve_spd",
                    expected: "a vector or matrix",
                    rank: rhs.ndim(),
                });
            }
        }
        for row in 0..order {
            let diagonal = self.l.data()[row * order + row];
            if diagonal <= tolerance::<T>() {
                return Err(AtlasLinalgError::NotPositiveDefinite { op: "solve_spd", index: row });
            }
        }

        let intermediate = solve_lower_triangular_with_op(&self.l, rhs, "solve_spd")?;

        solve_upper_triangular_with_op(self.l.view().transpose(), &intermediate, "solve_spd")
    }

    fn order(&self) -> AtlasLinalgResult<usize> {
        let [rows, columns] = self.l.shape() else {
            return Err(AtlasLinalgError::InvalidInputShape {
                op: "solve_spd",
                shape: self.l.shape().to_vec(),
                reason: "Cholesky lower factor must be a square matrix",
            });
        };

        if rows != columns {
            return Err(AtlasLinalgError::InvalidInputShape {
                op: "solve_spd",
                shape: self.l.shape().to_vec(),
                reason: "Cholesky lower factor must be square",
            });
        }

        validate_lower_triangular(&self.l, "solve_spd", "Cholesky lower", false)?;

        Ok(*rows)
    }
}

pub fn cholesky<'a, T, M>(matrix: M) -> AtlasLinalgResult<CholeskyFactorization<T>>
where
    T: Numeric + Float + 'a,
    M: Into<LinalgOperand<'a, T>>,
{
    let matrix = matrix.into();
    cholesky_operand(&matrix)
}

fn cholesky_operand<T>(matrix: &LinalgOperand<'_, T>) -> AtlasLinalgResult<CholeskyFactorization<T>>
where
    T: Numeric + Float,
{
    let (rows, cols) = validate_rank_two(matrix, "cholesky")?;

    if rows != cols {
        return Err(AtlasLinalgError::InvalidInputShape {
            op: "cholesky",
            shape: matrix.shape().to_vec(),
            reason: "Cholesky requires a square matrix",
        });
    }

    let n = rows;
    let tolerance = tolerance::<T>();
    let a = copy_matrix_row_major(matrix);
    validate_finite(&a, "cholesky")?;

    if !is_symmetric(&a, n, tolerance) {
        return Err(AtlasLinalgError::InvalidInputShape {
            op: "cholesky",
            shape: matrix.shape().to_vec(),
            reason: "Cholesky requires a symmetric matrix",
        });
    }

    let mut l = zero_matrix_data(n, n);

    for row in 0..n {
        let a_row = &a[row * n..(row + 1) * n];
        let (head, tail) = l.split_at_mut(row * n);
        let l_row = &mut tail[..n];

        for col in 0..row {
            let l_col = &head[col * n..(col + 1) * n];
            let value = a_row[col] - dot_slice(&l_row[..col], &l_col[..col]);

            l_row[col] = value / l_col[col];
        }

        let diagonal = a_row[row] - dot_slice(&l_row[..row], &l_row[..row]);

        if diagonal <= tolerance {
            return Err(AtlasLinalgError::NotPositiveDefinite { op: "cholesky", index: row });
        }

        l_row[row] = diagonal.sqrt();
    }

    Ok(CholeskyFactorization { l: NDArray::from_shape_vec([n, n], l)? })
}

/// Solves a rank-2 SPD system or matching rank-3 batches of SPD systems.
pub fn solve_spd<'a, 'b, T, M, R>(matrix: M, rhs: R) -> AtlasLinalgResult<NDArray<T>>
where
    T: Numeric + Float + 'a + 'b,
    M: Into<LinalgOperand<'a, T>>,
    R: Into<LinalgOperand<'b, T>>,
{
    let matrix = matrix.into();
    let rhs = rhs.into();

    if matrix.ndim() == 3 {
        solve_batched_spd(&matrix, &rhs)
    } else {
        cholesky_operand(&matrix)?.solve_operand(&rhs)
    }
}

fn solve_batched_spd<T>(
    matrix: &LinalgOperand<'_, T>,
    rhs: &LinalgOperand<'_, T>,
) -> AtlasLinalgResult<NDArray<T>>
where
    T: Numeric + Float,
{
    let [batch_count, rows, columns] = matrix.shape() else {
        unreachable!("batched SPD solve requires a rank-3 coefficient matrix");
    };
    let (batch_count, rows, columns) = (*batch_count, *rows, *columns);
    if rows != columns {
        return Err(AtlasLinalgError::InvalidInputShape {
            op: "cholesky",
            shape: matrix.shape().to_vec(),
            reason: "Cholesky requires a square matrix",
        });
    }

    let (rhs_columns, vector_rhs) = match rhs.shape() {
        [rhs_batches, rhs_rows] if *rhs_batches == batch_count && *rhs_rows == rows => (1, true),
        [rhs_batches, rhs_rows, rhs_columns]
            if *rhs_batches == batch_count && *rhs_rows == rows =>
        {
            (*rhs_columns, false)
        }
        [rhs_batches, ..]
            if (rhs.ndim() == 2 || rhs.ndim() == 3) && *rhs_batches != batch_count =>
        {
            return Err(AtlasLinalgError::ShapeMismatch {
                op: "solve_spd",
                left: matrix.shape().to_vec(),
                right: rhs.shape().to_vec(),
                reason: "batch dimensions must match",
            });
        }
        [..] if rhs.ndim() == 2 || rhs.ndim() == 3 => {
            return Err(AtlasLinalgError::ShapeMismatch {
                op: "solve_spd",
                left: matrix.shape().to_vec(),
                right: rhs.shape().to_vec(),
                reason: "right-hand side row count must match coefficient matrix row count",
            });
        }
        _ => {
            return Err(AtlasLinalgError::InvalidInputRank {
                op: "solve_spd",
                expected: "a rank-2 batched vector or rank-3 batched matrix",
                rank: rhs.ndim(),
            });
        }
    };
    let mut solutions = Vec::with_capacity(batch_count * rows * rhs_columns);

    for batch in 0..batch_count {
        let matrix = batched_matrix(matrix, batch, rows)?;
        let rhs = batched_rhs(rhs, batch, rows, rhs_columns, vector_rhs)?;
        let factor = cholesky_operand(&LinalgOperand::from(&matrix))?;
        let solution = factor.solve_operand(&LinalgOperand::from(&rhs))?;
        solutions.extend_from_slice(solution.data());
    }

    if vector_rhs {
        NDArray::from_shape_vec([batch_count, rows], solutions).map_err(Into::into)
    } else {
        NDArray::from_shape_vec([batch_count, rows, rhs_columns], solutions).map_err(Into::into)
    }
}

fn batched_matrix<T: Numeric>(
    matrix: &LinalgOperand<'_, T>,
    batch: usize,
    rows: usize,
) -> AtlasLinalgResult<NDArray<T>> {
    let mut values = Vec::with_capacity(rows * rows);
    for row in 0..rows {
        for column in 0..rows {
            values.push(
                matrix.data()[matrix.offset()
                    + batch * matrix.strides()[0]
                    + row * matrix.strides()[1]
                    + column * matrix.strides()[2]],
            );
        }
    }
    NDArray::from_shape_vec([rows, rows], values).map_err(Into::into)
}

fn batched_rhs<T: Numeric>(
    rhs: &LinalgOperand<'_, T>,
    batch: usize,
    rows: usize,
    columns: usize,
    vector_rhs: bool,
) -> AtlasLinalgResult<NDArray<T>> {
    let mut values = Vec::with_capacity(rows * columns);
    for row in 0..rows {
        for column in 0..columns {
            let offset = rhs.offset() + batch * rhs.strides()[0] + row * rhs.strides()[1];
            values.push(if vector_rhs {
                rhs.data()[offset]
            } else {
                rhs.data()[offset + column * rhs.strides()[2]]
            });
        }
    }
    if vector_rhs {
        NDArray::from_shape_vec([rows], values).map_err(Into::into)
    } else {
        NDArray::from_shape_vec([rows, columns], values).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::CholeskyFactorization;
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

        assert_eq!(factor.l().shape(), &[2, 2]);

        let reconstructed = matmul(factor.l(), factor.l().view().transpose()).unwrap();
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

    #[test]
    fn cholesky_solve_rejects_a_non_lower_factor() {
        let factor = CholeskyFactorization {
            l: NDArray::from_shape_vec([2, 2], vec![1.0_f64, 1.0, 0.0, 1.0]).unwrap(),
        };
        let rhs = NDArray::from_shape_vec([2], vec![1.0_f64, 2.0]).unwrap();

        assert!(matches!(
            factor.solve(&rhs),
            Err(AtlasLinalgError::InvalidFactor {
                op: "solve_spd",
                factor: "Cholesky lower",
                reason: "must be triangular",
            })
        ));
    }

    #[test]
    fn cholesky_solve_rejects_non_positive_diagonal_factors() {
        let factor = CholeskyFactorization {
            l: NDArray::from_shape_vec([2, 2], vec![1.0_f64, 0.0, 0.0, 0.0]).unwrap(),
        };
        let rhs = NDArray::from_shape_vec([2], vec![1.0_f64, 2.0]).unwrap();

        assert!(matches!(
            factor.solve(&rhs),
            Err(AtlasLinalgError::NotPositiveDefinite { op: "solve_spd", index: 1 })
        ));
    }
}
