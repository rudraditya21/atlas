use atlas_ndarray::{ArrayElement, OperandMetadata};

/// Copies one logical row from a rank-2 ndarray operand into `row`.
///
/// Callers validate the operand rank, row index, and destination width.
pub(crate) fn copy_logical_row<T, O>(operand: &O, row_index: usize, row: &mut [T])
where
    T: ArrayElement,
    O: OperandMetadata<T> + ?Sized,
{
    debug_assert_eq!(operand.ndim(), 2);
    debug_assert!(row_index < operand.shape()[0]);
    debug_assert_eq!(row.len(), operand.shape()[1]);

    let row_offset = operand.offset() + row_index * operand.strides()[0];
    for (column, value) in row.iter_mut().enumerate() {
        *value = operand.data()[row_offset + column * operand.strides()[1]];
    }
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::copy_logical_row;

    #[test]
    fn copies_contiguous_transposed_sliced_and_empty_rows_logically() {
        let contiguous = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let transposed = contiguous.view().transpose();
        let sliced = contiguous.view().slice([0, 1], [2, 2]).unwrap();
        let empty = NDArray::<i32>::from_shape_vec([2, 0], vec![]).unwrap();

        let mut row = [0; 3];
        copy_logical_row(&contiguous, 1, &mut row);
        assert_eq!(row, [3, 4, 5]);

        let mut row = [0; 2];
        copy_logical_row(&transposed, 2, &mut row);
        assert_eq!(row, [2, 5]);
        copy_logical_row(&sliced, 1, &mut row);
        assert_eq!(row, [4, 5]);

        copy_logical_row(&empty, 1, &mut []);
    }
}
