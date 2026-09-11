use atlas_ndarray::{ArrayView, NDArray, Numeric, checked_element_count};

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

/// Extracts the diagonal at `offset` from every matrix in a rank-3 batch.
pub fn batched_diag<'a, T, O>(matrices: O, offset: isize) -> AtlasLinalgResult<NDArray<T>>
where
    T: Numeric + 'a,
    O: Into<LinalgOperand<'a, T>>,
{
    let matrices = matrices.into();
    let [batches, rows, columns] = matrices.shape() else {
        return Err(AtlasLinalgError::InvalidInputRank {
            op: "batched_diag",
            expected: "a rank-3 [batch, rows, columns] array",
            rank: matrices.ndim(),
        });
    };
    let (row, column) = if offset >= 0 { (0, offset as usize) } else { (offset.unsigned_abs(), 0) };
    let length = rows.saturating_sub(row).min(columns.saturating_sub(column));
    let mut values = Vec::with_capacity(checked_element_count(&[*batches, length])?);

    for batch in 0..*batches {
        for index in 0..length {
            values.push(
                matrices.data()[matrices.offset()
                    + batch * matrices.strides()[0]
                    + (row + index) * matrices.strides()[1]
                    + (column + index) * matrices.strides()[2]],
            );
        }
    }

    Ok(NDArray::from_shape_vec([*batches, length], values)?)
}

pub fn diag_view<'a, T, O>(matrix: O, offset: isize) -> AtlasLinalgResult<ArrayView<'a, T>>
where
    T: Numeric + 'a,
    O: Into<LinalgOperand<'a, T>>,
{
    let matrix = matrix.into();

    if matrix.ndim() != 2 {
        return Err(AtlasLinalgError::InvalidInputRank {
            op: "diag_view",
            expected: "a 2-D array",
            rank: matrix.ndim(),
        });
    }

    matrix.into_view().diagonal(offset).map_err(Into::into)
}
