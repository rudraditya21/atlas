use atlas_ndarray::Numeric;

use crate::core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand};
use crate::internal::dense::matrix_ref;

pub fn trace<'a, T, O>(operand: O) -> AtlasLinalgResult<T>
where
    T: Numeric + 'a,
    O: Into<LinalgOperand<'a, T>>,
{
    let operand = operand.into();

    if operand.ndim() != 2 {
        return Err(AtlasLinalgError::InvalidInputRank {
            op: "trace",
            expected: "a 2-D array",
            rank: operand.ndim(),
        });
    }

    Ok(trace_matrix(matrix_ref(&operand)))
}

fn trace_matrix<T: Numeric>(matrix: crate::internal::dense::MatrixRef<'_, T>) -> T {
    let diagonal_len = matrix.rows.min(matrix.cols);

    if diagonal_len == 0 {
        return T::zero();
    }

    if matrix.is_row_major_contiguous() {
        let stride = matrix.cols + 1;
        let region = matrix.row_major_region();
        let mut total = T::zero();

        for index in 0..diagonal_len {
            total += region[index * stride];
        }

        total
    } else if matrix.is_col_major_contiguous() {
        let stride = matrix.rows + 1;
        let first_col = matrix.offset;
        let mut total = T::zero();

        for index in 0..diagonal_len {
            total += matrix.data[first_col + index * stride];
        }

        total
    } else {
        let mut total = T::zero();

        for index in 0..diagonal_len {
            total += matrix.value_at(index, index);
        }

        total
    }
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::trace;
    use crate::AtlasLinalgError;

    #[test]
    fn trace_supports_square_rectangular_and_empty_matrices() {
        let square = NDArray::from_shape_vec([2, 2], vec![1_i32, 2, 3, 4]).unwrap();
        let rectangular = NDArray::from_shape_vec([2, 3], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
        let empty = NDArray::from_shape_vec([0, 3], Vec::<i32>::new()).unwrap();

        assert_eq!(trace(&square).unwrap(), 5);
        assert_eq!(trace(&rectangular).unwrap(), 6);
        assert_eq!(trace(&empty).unwrap(), 0);
    }

    #[test]
    fn trace_rejects_non_matrix_inputs() {
        let vector = NDArray::from_shape_vec([3], vec![1_i32, 2, 3]).unwrap();
        let scalar = NDArray::from_shape_vec([], vec![7_i32]).unwrap();

        assert_eq!(
            trace(&vector).unwrap_err(),
            AtlasLinalgError::InvalidInputRank { op: "trace", expected: "a 2-D array", rank: 1 }
        );
        assert_eq!(
            trace(&scalar).unwrap_err(),
            AtlasLinalgError::InvalidInputRank { op: "trace", expected: "a 2-D array", rank: 0 }
        );
    }
}
