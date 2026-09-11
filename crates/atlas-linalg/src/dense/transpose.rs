use atlas_ndarray::{NDArray, Numeric, checked_element_count};

use crate::core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand};

/// Transposes every `[rows, columns]` matrix in a rank-3 `[batch, rows, columns]` operand.
pub fn batched_transpose<'a, T, O>(input: O) -> AtlasLinalgResult<NDArray<T>>
where
    T: Numeric + 'a,
    O: Into<LinalgOperand<'a, T>>,
{
    let input = input.into();
    let [batches, rows, columns] = input.shape() else {
        return Err(AtlasLinalgError::InvalidInputRank {
            op: "batched_transpose",
            expected: "a rank-3 [batch, rows, columns] array",
            rank: input.ndim(),
        });
    };
    let mut data = Vec::with_capacity(checked_element_count(&[*batches, *rows, *columns])?);

    for batch in 0..*batches {
        for column in 0..*columns {
            for row in 0..*rows {
                data.push(
                    input.data()[input.offset()
                        + batch * input.strides()[0]
                        + row * input.strides()[1]
                        + column * input.strides()[2]],
                );
            }
        }
    }

    Ok(NDArray::from_shape_vec([*batches, *columns, *rows], data)?)
}
