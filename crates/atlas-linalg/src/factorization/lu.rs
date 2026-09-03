use atlas_ndarray::{NDArray, Numeric};
use num_traits::Float;

use crate::core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand};
use crate::dense::triangular::{solve_lower_triangular_with_op, solve_upper_triangular_with_op};
use crate::internal::factorization::{
    copy_matrix_row_major, find_pivot_row, identity_matrix_data, swap_l_prefix_rows, swap_rows,
    tolerance, validate_rank_two, zero_matrix_data,
};

#[derive(Clone, Debug)]
pub struct LuFactorization<T: Numeric> {
    pub p: NDArray<T>,
    pub l: NDArray<T>,
    pub u: NDArray<T>,
}

impl<T: Numeric + Float> LuFactorization<T> {
    pub fn solve<'a, R>(&self, rhs: R) -> AtlasLinalgResult<NDArray<T>>
    where
        T: 'a,
        R: Into<LinalgOperand<'a, T>>,
    {
        let order = self.order("solve")?;
        let rhs = rhs.into();
        let (rhs_columns, vector_rhs) = match rhs.shape() {
            [rows] if *rows == order => (1, true),
            [rows, columns] if *rows == order => (*columns, false),
            [..] if rhs.ndim() == 1 || rhs.ndim() == 2 => {
                return Err(AtlasLinalgError::ShapeMismatch {
                    op: "solve",
                    left: vec![order, order],
                    right: rhs.shape().to_vec(),
                    reason: "right-hand side row count must match coefficient matrix row count",
                });
            }
            _ => {
                return Err(AtlasLinalgError::InvalidInputRank {
                    op: "solve",
                    expected: "a vector or matrix",
                    rank: rhs.ndim(),
                });
            }
        };
        let mut values = vec![T::zero(); order * rhs_columns];

        for row in 0..order {
            for column in 0..rhs_columns {
                let mut value = T::zero();
                for source_row in 0..order {
                    value += self.p.data()[row * order + source_row]
                        * rhs_value(&rhs, source_row, column);
                }
                values[row * rhs_columns + column] = value;
            }
        }

        let permuted_rhs = if vector_rhs {
            NDArray::from_shape_vec([order], values)?
        } else {
            NDArray::from_shape_vec([order, rhs_columns], values)?
        };
        let intermediate = solve_lower_triangular_with_op(&self.l, &permuted_rhs, "solve")?;

        solve_upper_triangular_with_op(&self.u, &intermediate, "solve")
    }

    pub fn det(&self) -> AtlasLinalgResult<T> {
        let order = self.order("det")?;
        let mut determinant = T::one();

        for index in 0..order {
            determinant = determinant * self.u.data()[index * order + index];
        }

        if self.permutation_is_odd(order)? { Ok(-determinant) } else { Ok(determinant) }
    }

    pub fn slogdet(&self) -> AtlasLinalgResult<(T, T)> {
        let order = self.order("slogdet")?;
        let mut sign = if self.permutation_is_odd(order)? { -T::one() } else { T::one() };
        let mut log_abs_det = T::zero();

        for index in 0..order {
            let diagonal = self.u.data()[index * order + index];
            if diagonal < T::zero() {
                sign = -sign;
            }
            log_abs_det += diagonal.abs().ln();
        }

        Ok((sign, log_abs_det))
    }

    fn order(&self, op: &'static str) -> AtlasLinalgResult<usize> {
        let [rows, columns] = self.u.shape() else {
            return Err(AtlasLinalgError::InvalidInputShape {
                op,
                shape: self.u.shape().to_vec(),
                reason: "LU upper factor must be a square matrix",
            });
        };

        if rows != columns || self.p.shape() != [*rows, *rows] || self.l.shape() != [*rows, *rows] {
            return Err(AtlasLinalgError::InvalidInputShape {
                op,
                shape: self.u.shape().to_vec(),
                reason: "LU factors must be square matrices of the same order",
            });
        }

        Ok(*rows)
    }

    fn permutation_is_odd(&self, order: usize) -> AtlasLinalgResult<bool> {
        let mut permutation = Vec::with_capacity(order);
        let mut used_columns = vec![false; order];

        for row in 0..order {
            let mut selected_column = None;

            for column in 0..order {
                let value = self.p.data()[row * order + column];
                if value == T::one() {
                    if selected_column.replace(column).is_some() {
                        return Err(invalid_permutation_error(self.p.shape()));
                    }
                } else if !value.is_zero() {
                    return Err(invalid_permutation_error(self.p.shape()));
                }
            }

            let Some(column) = selected_column else {
                return Err(invalid_permutation_error(self.p.shape()));
            };
            if used_columns[column] {
                return Err(invalid_permutation_error(self.p.shape()));
            }

            used_columns[column] = true;
            permutation.push(column);
        }

        let inversions = (0..order)
            .flat_map(|row| ((row + 1)..order).map(move |next_row| (row, next_row)))
            .filter(|&(row, next_row)| permutation[row] > permutation[next_row])
            .count();

        Ok(inversions % 2 == 1)
    }
}

fn invalid_permutation_error(shape: &[usize]) -> AtlasLinalgError {
    AtlasLinalgError::InvalidInputShape {
        op: "det",
        shape: shape.to_vec(),
        reason: "LU permutation factor must be a permutation matrix",
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

pub fn lu<'a, T, M>(matrix: M) -> AtlasLinalgResult<LuFactorization<T>>
where
    T: Numeric + Float + 'a,
    M: Into<LinalgOperand<'a, T>>,
{
    let matrix = matrix.into();
    let (rows, cols) = validate_rank_two(&matrix, "lu")?;

    if rows != cols {
        return Err(AtlasLinalgError::InvalidInputShape {
            op: "lu",
            shape: matrix.shape().to_vec(),
            reason: "LU requires a square matrix",
        });
    }

    let n = rows;
    let tolerance = tolerance::<T>();
    let mut a = copy_matrix_row_major(&matrix);
    let mut p = identity_matrix_data(n);
    let mut l = zero_matrix_data(n, n);

    for diagonal in 0..n {
        l[diagonal * n + diagonal] = T::one();
    }

    for k in 0..n {
        let pivot_row = find_pivot_row(&a, n, k);
        let pivot_value = a[pivot_row * n + k].abs();

        if pivot_value <= tolerance {
            return Err(AtlasLinalgError::SingularMatrix { op: "lu", pivot: k });
        }

        if pivot_row != k {
            swap_rows(&mut a, n, k, pivot_row);
            swap_rows(&mut p, n, k, pivot_row);
            swap_l_prefix_rows(&mut l, n, k, pivot_row, k);
        }

        let pivot_diagonal = a[k * n + k];

        for row in (k + 1)..n {
            let row_start = row * n;
            let (head, tail) = a.split_at_mut(row_start);
            let pivot_row = &head[k * n..(k + 1) * n];
            let row_slice = &mut tail[..n];
            let factor = row_slice[k] / pivot_diagonal;

            l[row * n + k] = factor;
            row_slice[k] = T::zero();

            for (value, &pivot_value) in
                row_slice[k + 1..].iter_mut().zip(pivot_row[k + 1..].iter())
            {
                *value -= factor * pivot_value;
            }
        }
    }

    Ok(LuFactorization {
        p: NDArray::from_shape_vec([n, n], p)?,
        l: NDArray::from_shape_vec([n, n], l)?,
        u: NDArray::from_shape_vec([n, n], a)?,
    })
}

pub fn solve<'a, 'b, T, M, R>(matrix: M, rhs: R) -> AtlasLinalgResult<NDArray<T>>
where
    T: Numeric + Float + 'a + 'b,
    M: Into<LinalgOperand<'a, T>>,
    R: Into<LinalgOperand<'b, T>>,
{
    lu(matrix)?.solve(rhs)
}

pub fn inverse<'a, T, M>(matrix: M) -> AtlasLinalgResult<NDArray<T>>
where
    T: Numeric + Float + 'a,
    M: Into<LinalgOperand<'a, T>>,
{
    let factorization = lu(matrix)?;
    let identity = NDArray::eye(factorization.u.shape()[0])?;

    factorization.solve(&identity)
}

pub fn det<'a, T, M>(matrix: M) -> AtlasLinalgResult<T>
where
    T: Numeric + Float + 'a,
    M: Into<LinalgOperand<'a, T>>,
{
    lu(matrix)?.det()
}

pub fn slogdet<'a, T, M>(matrix: M) -> AtlasLinalgResult<(T, T)>
where
    T: Numeric + Float + 'a,
    M: Into<LinalgOperand<'a, T>>,
{
    lu(matrix)?.slogdet()
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use crate::{AtlasLinalgError, lu, matmul};

    fn assert_close_slice(actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() <= tolerance);
        }
    }

    #[test]
    fn lu_returns_permutation_lower_and_upper_factors() {
        let matrix = NDArray::from_shape_vec([2, 2], vec![0.0_f64, 2.0, 1.0, 3.0]).unwrap();
        let factors = lu(&matrix).unwrap();

        assert_eq!(factors.p.shape(), &[2, 2]);
        assert_eq!(factors.l.shape(), &[2, 2]);
        assert_eq!(factors.u.shape(), &[2, 2]);

        let pa = matmul(&factors.p, &matrix).unwrap();
        let lu_product = matmul(&factors.l, &factors.u).unwrap();

        assert_close_slice(pa.data(), lu_product.data(), 1e-10);
    }

    #[test]
    fn lu_reports_expected_validation_errors() {
        let vector = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 3.0]).unwrap();
        let singular = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 2.0, 2.0, 4.0]).unwrap();

        assert!(matches!(
            lu(&vector).unwrap_err(),
            AtlasLinalgError::InvalidInputRank { op: "lu", .. }
        ));
        assert!(matches!(
            lu(&singular).unwrap_err(),
            AtlasLinalgError::SingularMatrix { op: "lu", .. }
        ));
    }
}
