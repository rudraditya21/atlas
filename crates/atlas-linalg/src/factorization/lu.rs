use atlas_ndarray::{NDArray, Numeric};
use num_traits::Float;

use crate::core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand};

use super::{
    copy_matrix_row_major, find_pivot_row, identity_matrix_data, swap_l_prefix_rows, swap_rows,
    tolerance, validate_rank_two, zero_matrix_data,
};

#[derive(Clone, Debug)]
pub struct LuFactorization<T: Numeric> {
    pub p: NDArray<T>,
    pub l: NDArray<T>,
    pub u: NDArray<T>,
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
