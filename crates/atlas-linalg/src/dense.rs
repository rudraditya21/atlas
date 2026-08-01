use atlas_ndarray::{NDArray, Numeric};

use crate::error::{AtlasLinalgError, AtlasLinalgResult};

pub fn dot<T: Numeric>(lhs: &NDArray<T>, rhs: &NDArray<T>) -> AtlasLinalgResult<T> {
    match (lhs.shape(), rhs.shape()) {
        ([lhs_len], [rhs_len]) if lhs_len == rhs_len => {
            let mut total = T::zero();

            for index in 0..*lhs_len {
                total += lhs.data()[index] * rhs.data()[index];
            }

            Ok(total)
        }
        ([..], [..]) if lhs.ndim() == 1 && rhs.ndim() == 1 => {
            Err(AtlasLinalgError::ShapeMismatch {
                op: "dot",
                left: lhs.shape().to_vec(),
                right: rhs.shape().to_vec(),
                reason: "vector lengths must match",
            })
        }
        _ => Err(AtlasLinalgError::InvalidOperandRank {
            op: "dot",
            left: lhs.ndim(),
            right: rhs.ndim(),
        }),
    }
}

pub fn matmul<T: Numeric>(lhs: &NDArray<T>, rhs: &NDArray<T>) -> AtlasLinalgResult<NDArray<T>> {
    match (lhs.ndim(), rhs.ndim()) {
        (1, 1) => matmul_vector_vector(lhs, rhs),
        (1, 2) => matmul_vector_matrix(lhs, rhs),
        (2, 1) => matmul_matrix_vector(lhs, rhs),
        (2, 2) => matmul_matrix_matrix(lhs, rhs),
        _ => Err(AtlasLinalgError::InvalidOperandRank {
            op: "matmul",
            left: lhs.ndim(),
            right: rhs.ndim(),
        }),
    }
}

fn matmul_vector_vector<T: Numeric>(
    lhs: &NDArray<T>,
    rhs: &NDArray<T>,
) -> AtlasLinalgResult<NDArray<T>> {
    let value = dot(lhs, rhs)?;
    wrap_scalar(value)
}

fn matmul_vector_matrix<T: Numeric>(
    lhs: &NDArray<T>,
    rhs: &NDArray<T>,
) -> AtlasLinalgResult<NDArray<T>> {
    let lhs_len = lhs.shape()[0];
    let rhs_rows = rhs.shape()[0];
    let rhs_cols = rhs.shape()[1];

    if lhs_len != rhs_rows {
        return Err(AtlasLinalgError::ShapeMismatch {
            op: "matmul",
            left: lhs.shape().to_vec(),
            right: rhs.shape().to_vec(),
            reason: "vector length must match matrix row count",
        });
    }

    let mut data = Vec::with_capacity(rhs_cols);

    for col in 0..rhs_cols {
        let mut total = T::zero();

        for k in 0..lhs_len {
            total += lhs.data()[k] * *rhs.get(&[k, col])?;
        }

        data.push(total);
    }

    Ok(NDArray::from_shape_vec([rhs_cols], data)?)
}

fn matmul_matrix_vector<T: Numeric>(
    lhs: &NDArray<T>,
    rhs: &NDArray<T>,
) -> AtlasLinalgResult<NDArray<T>> {
    let lhs_rows = lhs.shape()[0];
    let lhs_cols = lhs.shape()[1];
    let rhs_len = rhs.shape()[0];

    if lhs_cols != rhs_len {
        return Err(AtlasLinalgError::ShapeMismatch {
            op: "matmul",
            left: lhs.shape().to_vec(),
            right: rhs.shape().to_vec(),
            reason: "matrix column count must match vector length",
        });
    }

    let mut data = Vec::with_capacity(lhs_rows);

    for row in 0..lhs_rows {
        let mut total = T::zero();

        for k in 0..lhs_cols {
            total += *lhs.get(&[row, k])? * rhs.data()[k];
        }

        data.push(total);
    }

    Ok(NDArray::from_shape_vec([lhs_rows], data)?)
}

fn matmul_matrix_matrix<T: Numeric>(
    lhs: &NDArray<T>,
    rhs: &NDArray<T>,
) -> AtlasLinalgResult<NDArray<T>> {
    let lhs_rows = lhs.shape()[0];
    let lhs_cols = lhs.shape()[1];
    let rhs_rows = rhs.shape()[0];
    let rhs_cols = rhs.shape()[1];

    if lhs_cols != rhs_rows {
        return Err(AtlasLinalgError::ShapeMismatch {
            op: "matmul",
            left: lhs.shape().to_vec(),
            right: rhs.shape().to_vec(),
            reason: "left matrix column count must match right matrix row count",
        });
    }

    let mut data = vec![T::zero(); lhs_rows * rhs_cols];

    let lhs_data = lhs.data();
    let rhs_data = rhs.data();

    for row in 0..lhs_rows {
        let lhs_row_offset = row * lhs_cols;

        for k in 0..lhs_cols {
            let lhs_value = lhs_data[lhs_row_offset + k];
            let rhs_row_offset = k * rhs_cols;
            let out_row_offset = row * rhs_cols;

            for col in 0..rhs_cols {
                data[out_row_offset + col] += lhs_value * rhs_data[rhs_row_offset + col];
            }
        }
    }

    Ok(NDArray::from_shape_vec([lhs_rows, rhs_cols], data)?)
}

fn wrap_scalar<T: Numeric>(value: T) -> AtlasLinalgResult<NDArray<T>> {
    Ok(NDArray::from_shape_vec([], vec![value])?)
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use crate::{dot, error::AtlasLinalgError, matmul};

    #[test]
    fn dot_rejects_non_vector_inputs_and_mismatched_lengths() {
        let lhs = NDArray::from_shape_vec([3], vec![1_i32, 2, 3]).unwrap();
        let rhs = NDArray::from_shape_vec([2], vec![4_i32, 5]).unwrap();
        let matrix = NDArray::from_shape_vec([1, 3], vec![1_i32, 2, 3]).unwrap();

        assert!(matches!(
            dot(&lhs, &rhs).unwrap_err(),
            AtlasLinalgError::ShapeMismatch { op: "dot", .. }
        ));
        assert!(matches!(
            dot(&lhs, &matrix).unwrap_err(),
            AtlasLinalgError::InvalidOperandRank { op: "dot", .. }
        ));
    }

    #[test]
    fn matmul_supports_vector_and_matrix_operands() {
        let lhs_vec = NDArray::from_shape_vec([3], vec![1_i32, 2, 3]).unwrap();
        let rhs_vec = NDArray::from_shape_vec([3], vec![4_i32, 5, 6]).unwrap();
        let matrix = NDArray::from_shape_vec([3, 2], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
        let left_matrix = NDArray::from_shape_vec([2, 3], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
        let right_matrix = NDArray::from_shape_vec([3, 2], vec![7_i32, 8, 9, 10, 11, 12]).unwrap();

        assert_eq!(dot(&lhs_vec, &rhs_vec).unwrap(), 32);
        assert_eq!(matmul(&lhs_vec, &rhs_vec).unwrap().data(), &[32]);
        assert_eq!(matmul(&lhs_vec, &matrix).unwrap().data(), &[22, 28]);
        assert_eq!(matmul(&left_matrix, &rhs_vec).unwrap().data(), &[32, 77]);
        assert_eq!(
            matmul(&left_matrix, &right_matrix).unwrap().shape(),
            &[2, 2]
        );
        assert_eq!(
            matmul(&left_matrix, &right_matrix).unwrap().data(),
            &[58, 64, 139, 154]
        );
    }
}
