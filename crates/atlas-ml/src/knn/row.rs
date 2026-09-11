use atlas_ndarray::OperandMetadata;

pub(crate) fn copy_row<O>(operand: &O, row_index: usize, row: &mut [f64])
where
    O: OperandMetadata<f64> + ?Sized,
{
    let row_offset = operand.offset() + row_index * operand.strides()[0];
    for (feature_index, value) in row.iter_mut().enumerate() {
        *value = operand.data()[row_offset + feature_index * operand.strides()[1]];
    }
}
