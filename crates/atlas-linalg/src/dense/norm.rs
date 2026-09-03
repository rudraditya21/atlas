use atlas_ndarray::{AtlasNdError, Numeric};
use num_traits::ToPrimitive;

use crate::core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand};
use crate::internal::dense::{matrix_ref, vector_ref};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixNorm {
    Frobenius,
    L1,
    Infinity,
}

pub fn norm<'a, T, O>(operand: O) -> AtlasLinalgResult<f64>
where
    T: Numeric + ToPrimitive + 'a,
    O: Into<LinalgOperand<'a, T>>,
{
    let operand = operand.into();

    match operand.ndim() {
        1 => vector_norm(vector_ref(&operand)),
        2 => matrix_frobenius_norm(matrix_ref(&operand)),
        rank => Err(AtlasLinalgError::InvalidInputRank {
            op: "norm",
            expected: "a 1-D or 2-D array",
            rank,
        }),
    }
}

pub fn matrix_norm<'a, T, O>(matrix: O, order: MatrixNorm) -> AtlasLinalgResult<f64>
where
    T: Numeric + ToPrimitive + 'a,
    O: Into<LinalgOperand<'a, T>>,
{
    let matrix = matrix.into();

    if matrix.ndim() != 2 {
        return Err(AtlasLinalgError::InvalidInputRank {
            op: "matrix_norm",
            expected: "a 2-D array",
            rank: matrix.ndim(),
        });
    }

    let matrix = matrix_ref(&matrix);

    match order {
        MatrixNorm::Frobenius => matrix_frobenius_norm(matrix),
        MatrixNorm::L1 => matrix_l1_norm(matrix),
        MatrixNorm::Infinity => matrix_infinity_norm(matrix),
    }
}

fn vector_norm<T: Numeric + ToPrimitive>(
    vector: crate::internal::dense::VectorRef<'_, T>,
) -> AtlasLinalgResult<f64> {
    if vector.is_contiguous() {
        vector
            .contiguous_slice()
            .iter()
            .try_fold(0.0_f64, |total, &value| {
                let value =
                    value.to_f64().ok_or(AtlasNdError::NumericConversionFailed { op: "norm" })?;

                Ok::<f64, AtlasLinalgError>(total + value * value)
            })
            .map(f64::sqrt)
    } else {
        let mut total = 0.0_f64;

        for index in 0..vector.len {
            let value = vector
                .value_at(index)
                .to_f64()
                .ok_or(AtlasNdError::NumericConversionFailed { op: "norm" })?;
            total += value * value;
        }

        Ok(total.sqrt())
    }
}

fn matrix_frobenius_norm<T: Numeric + ToPrimitive>(
    matrix: crate::internal::dense::MatrixRef<'_, T>,
) -> AtlasLinalgResult<f64> {
    let total = if matrix.is_row_major_contiguous() {
        matrix.row_major_region().iter().try_fold(0.0_f64, |total, &value| {
            let value =
                value.to_f64().ok_or(AtlasNdError::NumericConversionFailed { op: "norm" })?;

            Ok::<f64, AtlasLinalgError>(total + value * value)
        })?
    } else if matrix.is_col_major_contiguous() {
        let mut total = 0.0_f64;

        for col in 0..matrix.cols {
            total =
                matrix.contiguous_col_slice(col).iter().try_fold(total, |col_total, &value| {
                    let value = value
                        .to_f64()
                        .ok_or(AtlasNdError::NumericConversionFailed { op: "norm" })?;

                    Ok::<f64, AtlasLinalgError>(col_total + value * value)
                })?;
        }

        total
    } else {
        let mut total = 0.0_f64;

        for row in 0..matrix.rows {
            for col in 0..matrix.cols {
                let value = matrix
                    .value_at(row, col)
                    .to_f64()
                    .ok_or(AtlasNdError::NumericConversionFailed { op: "norm" })?;
                total += value * value;
            }
        }

        total
    };

    Ok(total.sqrt())
}

fn matrix_l1_norm<T: Numeric + ToPrimitive>(
    matrix: crate::internal::dense::MatrixRef<'_, T>,
) -> AtlasLinalgResult<f64> {
    let mut maximum = 0.0;

    for column in 0..matrix.cols {
        let mut total = 0.0;
        for row in 0..matrix.rows {
            total += matrix
                .value_at(row, column)
                .to_f64()
                .ok_or(AtlasNdError::NumericConversionFailed { op: "matrix_norm" })?
                .abs();
        }
        if total.is_nan() {
            return Ok(f64::NAN);
        }
        maximum = maximum.max(total);
    }

    Ok(maximum)
}

fn matrix_infinity_norm<T: Numeric + ToPrimitive>(
    matrix: crate::internal::dense::MatrixRef<'_, T>,
) -> AtlasLinalgResult<f64> {
    let mut maximum = 0.0;

    for row in 0..matrix.rows {
        let mut total = 0.0;
        for column in 0..matrix.cols {
            total += matrix
                .value_at(row, column)
                .to_f64()
                .ok_or(AtlasNdError::NumericConversionFailed { op: "matrix_norm" })?
                .abs();
        }
        if total.is_nan() {
            return Ok(f64::NAN);
        }
        maximum = maximum.max(total);
    }

    Ok(maximum)
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::norm;
    use crate::AtlasLinalgError;

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() <= 1.0e-10);
    }

    #[test]
    fn norm_supports_vectors_matrices_and_empty_inputs() {
        let vector = NDArray::from_shape_vec([2], vec![3.0_f64, 4.0]).unwrap();
        let matrix = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        let integer_vector = NDArray::from_shape_vec([2], vec![5_i32, 12]).unwrap();
        let empty_vector = NDArray::from_shape_vec([0], Vec::<f64>::new()).unwrap();
        let empty_matrix = NDArray::from_shape_vec([0, 0], Vec::<f64>::new()).unwrap();

        assert_close(norm(&vector).unwrap(), 5.0);
        assert_close(norm(&matrix).unwrap(), 30.0_f64.sqrt());
        assert_close(norm(&integer_vector).unwrap(), 13.0);
        assert_close(norm(&empty_vector).unwrap(), 0.0);
        assert_close(norm(&empty_matrix).unwrap(), 0.0);
    }

    #[test]
    fn norm_rejects_ranks_outside_v0_scope() {
        let scalar = NDArray::from_shape_vec([], vec![7.0_f64]).unwrap();
        let tensor = NDArray::<f64>::zeros([1, 1, 1]).unwrap();

        assert_eq!(
            norm(&scalar).unwrap_err(),
            AtlasLinalgError::InvalidInputRank {
                op: "norm",
                expected: "a 1-D or 2-D array",
                rank: 0,
            }
        );
        assert_eq!(
            norm(&tensor).unwrap_err(),
            AtlasLinalgError::InvalidInputRank {
                op: "norm",
                expected: "a 1-D or 2-D array",
                rank: 3,
            }
        );
    }
}
