use atlas_ndarray::{NDArray, Numeric};

use crate::core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand};

pub fn diag<'a, T, O>(matrix: O, offset: isize) -> AtlasLinalgResult<NDArray<T>>
where
    T: Numeric + 'a,
    O: Into<LinalgOperand<'a, T>>,
{
    let matrix = matrix.into();

    if matrix.ndim() != 2 {
        return Err(AtlasLinalgError::InvalidInputRank {
            op: "diag",
            expected: "a 2-D array",
            rank: matrix.ndim(),
        });
    }

    let (row, column) = if offset >= 0 { (0, offset as usize) } else { (offset.unsigned_abs(), 0) };
    let length =
        matrix.shape()[0].saturating_sub(row).min(matrix.shape()[1].saturating_sub(column));
    let values = (0..length)
        .map(|index| {
            matrix.data()[matrix.offset()
                + (row + index) * matrix.strides()[0]
                + (column + index) * matrix.strides()[1]]
        })
        .collect();

    NDArray::from_shape_vec([length], values).map_err(Into::into)
}
