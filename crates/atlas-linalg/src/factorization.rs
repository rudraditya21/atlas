use atlas_ndarray::{NDArray, Numeric};
use num_traits::Float;

use crate::{
    error::{AtlasLinalgError, AtlasLinalgResult},
    operand::LinalgOperand,
};

#[derive(Clone, Debug)]
pub struct LuFactorization<T: Numeric> {
    pub p: NDArray<T>,
    pub l: NDArray<T>,
    pub u: NDArray<T>,
}

#[derive(Clone, Debug)]
pub struct QrFactorization<T: Numeric> {
    pub q: NDArray<T>,
    pub r: NDArray<T>,
}

#[derive(Clone, Debug)]
pub struct CholeskyFactorization<T: Numeric> {
    pub l: NDArray<T>,
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

        for row in (k + 1)..n {
            let factor = a[row * n + k] / a[k * n + k];
            l[row * n + k] = factor;
            a[row * n + k] = T::zero();

            for col in (k + 1)..n {
                a[row * n + col] = a[row * n + col] - factor * a[k * n + col];
            }
        }
    }

    Ok(LuFactorization {
        p: NDArray::from_shape_vec([n, n], p)?,
        l: NDArray::from_shape_vec([n, n], l)?,
        u: NDArray::from_shape_vec([n, n], a)?,
    })
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
                v[row] = v[row] - projection * q_columns[row * cols + prior];
            }
        }

        let norm = vector_norm(&v);

        if norm <= tolerance {
            return Err(AtlasLinalgError::RankDeficientMatrix {
                op: "qr",
                column: col,
            });
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
                value = value - l[row * n + inner] * l[col * n + inner];
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

    Ok(CholeskyFactorization {
        l: NDArray::from_shape_vec([n, n], l)?,
    })
}

fn validate_rank_two<T: Numeric>(
    operand: &LinalgOperand<'_, T>,
    op: &'static str,
) -> AtlasLinalgResult<(usize, usize)> {
    if operand.ndim() != 2 {
        return Err(AtlasLinalgError::InvalidInputRank {
            op,
            expected: "rank-2 matrix",
            rank: operand.ndim(),
        });
    }

    Ok((operand.shape()[0], operand.shape()[1]))
}

fn copy_matrix_row_major<T: Numeric>(operand: &LinalgOperand<'_, T>) -> Vec<T> {
    let rows = operand.shape()[0];
    let cols = operand.shape()[1];
    let row_stride = operand.strides()[0];
    let col_stride = operand.strides()[1];
    let offset = operand.offset();
    let data = operand.data();
    let mut copied = Vec::with_capacity(rows * cols);

    for row in 0..rows {
        for col in 0..cols {
            copied.push(data[offset + row * row_stride + col * col_stride]);
        }
    }

    copied
}

fn zero_matrix_data<T: Numeric>(rows: usize, cols: usize) -> Vec<T> {
    vec![T::zero(); rows * cols]
}

fn identity_matrix_data<T: Numeric>(n: usize) -> Vec<T> {
    let mut data = zero_matrix_data(n, n);

    for index in 0..n {
        data[index * n + index] = T::one();
    }

    data
}

fn find_pivot_row<T: Float>(matrix: &[T], n: usize, pivot_col: usize) -> usize {
    let mut pivot_row = pivot_col;
    let mut pivot_value = matrix[pivot_col * n + pivot_col].abs();

    for candidate in (pivot_col + 1)..n {
        let value = matrix[candidate * n + pivot_col].abs();

        if value > pivot_value {
            pivot_value = value;
            pivot_row = candidate;
        }
    }

    pivot_row
}

fn swap_rows<T>(matrix: &mut [T], cols: usize, left: usize, right: usize) {
    for col in 0..cols {
        matrix.swap(left * cols + col, right * cols + col);
    }
}

fn swap_l_prefix_rows<T>(matrix: &mut [T], cols: usize, left: usize, right: usize, end: usize) {
    for col in 0..end {
        matrix.swap(left * cols + col, right * cols + col);
    }
}

fn extract_column<T: Copy>(matrix: &[T], rows: usize, cols: usize, column: usize) -> Vec<T> {
    let mut values = Vec::with_capacity(rows);

    for row in 0..rows {
        values.push(matrix[row * cols + column]);
    }

    values
}

fn column_from_storage<T: Copy>(matrix: &[T], rows: usize, cols: usize, column: usize) -> Vec<T> {
    let mut values = Vec::with_capacity(rows);

    for row in 0..rows {
        values.push(matrix[row * cols + column]);
    }

    values
}

fn dot_slice<T: Numeric>(lhs: &[T], rhs: &[T]) -> T {
    let mut total = T::zero();

    for index in 0..lhs.len() {
        total += lhs[index] * rhs[index];
    }

    total
}

fn vector_norm<T: Numeric + Float>(values: &[T]) -> T {
    dot_slice(values, values).sqrt()
}

fn is_symmetric<T: Float>(matrix: &[T], n: usize, tolerance: T) -> bool {
    for row in 0..n {
        for col in 0..row {
            let diff = (matrix[row * n + col] - matrix[col * n + row]).abs();

            if diff > tolerance {
                return false;
            }
        }
    }

    true
}

fn tolerance<T: Float>() -> T {
    T::epsilon().sqrt()
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use crate::{AtlasLinalgError, cholesky, lu, matmul, qr};

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
    fn cholesky_returns_lower_triangular_factor() {
        let matrix = NDArray::from_shape_vec([2, 2], vec![4.0_f64, 2.0, 2.0, 3.0]).unwrap();
        let factor = cholesky(&matrix).unwrap();

        assert_eq!(factor.l.shape(), &[2, 2]);

        let reconstructed = matmul(&factor.l, factor.l.view().transpose()).unwrap();
        assert_close_slice(reconstructed.data(), matrix.data(), 1e-10);
    }

    #[test]
    fn factorizations_return_structured_errors() {
        let vector = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 3.0]).unwrap();
        let wide = NDArray::from_shape_vec([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let non_spd = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 2.0, 2.0, 1.0]).unwrap();

        assert!(matches!(
            lu(&vector).unwrap_err(),
            AtlasLinalgError::InvalidInputRank { op: "lu", .. }
        ));
        assert!(matches!(
            qr(&wide).unwrap_err(),
            AtlasLinalgError::InvalidInputShape { op: "qr", .. }
        ));
        assert!(matches!(
            cholesky(&non_spd).unwrap_err(),
            AtlasLinalgError::NotPositiveDefinite { op: "cholesky", .. }
        ));
    }
}
