use atlas_ndarray::{NDArray, Numeric};
use num_traits::Float;

use crate::core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand};
use crate::internal::factorization::{
    copy_matrix_row_major, dot_slice, is_symmetric, tolerance, validate_rank_two, zero_matrix_data,
};

#[derive(Clone, Debug)]
pub struct CholeskyFactorization<T: Numeric> {
    pub l: NDArray<T>,
}

impl<T: Numeric + Float> CholeskyFactorization<T> {
    pub fn solve<'a, R>(&self, rhs: R) -> AtlasLinalgResult<NDArray<T>>
    where
        T: 'a,
        R: Into<LinalgOperand<'a, T>>,
    {
        let order = self.order()?;
        let rhs = rhs.into();
        let (rhs_columns, vector_rhs) = match rhs.shape() {
            [rows] if *rows == order => (1, true),
            [rows, columns] if *rows == order => (*columns, false),
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
        };
        let mut values = vec![T::zero(); order * rhs_columns];

        for row in 0..order {
            let diagonal = self.l.data()[row * order + row];
            if diagonal <= tolerance::<T>() {
                return Err(AtlasLinalgError::NotPositiveDefinite { op: "solve_spd", index: row });
            }

            for column in 0..rhs_columns {
                let mut value = rhs_value(&rhs, row, column);
                for previous_row in 0..row {
                    value -= self.l.data()[row * order + previous_row]
                        * values[previous_row * rhs_columns + column];
                }
                values[row * rhs_columns + column] = value / diagonal;
            }
        }

        for row in (0..order).rev() {
            let diagonal = self.l.data()[row * order + row];

            for column in 0..rhs_columns {
                let mut value = values[row * rhs_columns + column];
                for next_row in (row + 1)..order {
                    value -= self.l.data()[next_row * order + row]
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

        Ok(*rows)
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

pub fn solve_spd<'a, 'b, T, M, R>(matrix: M, rhs: R) -> AtlasLinalgResult<NDArray<T>>
where
    T: Numeric + Float + 'a + 'b,
    M: Into<LinalgOperand<'a, T>>,
    R: Into<LinalgOperand<'b, T>>,
{
    cholesky(matrix)?.solve(rhs)
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
