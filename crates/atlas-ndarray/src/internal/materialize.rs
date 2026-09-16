use crate::{
    ArrayElement, NDArray, OperandMetadata,
    internal::{layout::is_contiguous_layout, shape::element_count},
};

pub(crate) fn materialize_contiguous_array<T, O>(operand: &O) -> NDArray<T>
where
    T: ArrayElement,
    O: OperandMetadata<T> + ?Sized,
{
    let data = if is_contiguous_layout(operand.shape(), operand.strides()) {
        operand
            .dense_slice()
            .expect("contiguous operands always expose a dense storage slice")
            .to_vec()
    } else {
        let mut data = Vec::with_capacity(element_count(operand.shape()));
        for span in super::logical_span_iter(
            operand.data(),
            operand.offset(),
            operand.shape(),
            operand.strides(),
        ) {
            data.extend_from_slice(span);
        }
        data
    };

    NDArray::from_row_major_parts(operand.shape().to_vec(), data)
        .expect("valid operands always materialize into valid owned arrays")
}

#[cfg(test)]
mod tests {
    use crate::{NDArray, internal::materialize_contiguous_array};

    #[test]
    fn materialize_contiguous_array_reuses_dense_storage_when_available() {
        let array = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let owned = materialize_contiguous_array(&array);

        assert_eq!(owned.shape(), &[2, 3]);
        assert_eq!(owned.strides(), &[3, 1]);
        assert_eq!(owned.data(), array.data());
    }

    #[test]
    fn materialize_contiguous_array_copies_strided_views_in_logical_order() {
        let array = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let view = array.view().transpose();
        let owned = materialize_contiguous_array(&view);

        assert_eq!(owned.shape(), &[3, 2]);
        assert_eq!(owned.strides(), &[2, 1]);
        assert_eq!(owned.data(), &[0, 3, 1, 4, 2, 5]);
        assert!(owned.is_contiguous());
    }

    #[test]
    fn materialize_contiguous_array_preserves_scalar_and_empty_shapes() {
        let scalar = NDArray::from_shape_vec([], vec![7_i32]).unwrap();
        let empty = NDArray::<i32>::from_shape_vec([2, 0, 3], Vec::new()).unwrap();

        let scalar_owned = materialize_contiguous_array(&scalar.view());
        let empty_owned = materialize_contiguous_array(&empty.view());

        assert_eq!(scalar_owned.shape(), &[] as &[usize]);
        assert_eq!(scalar_owned.strides(), &[] as &[usize]);
        assert_eq!(scalar_owned.data(), &[7]);

        assert_eq!(empty_owned.shape(), &[2, 0, 3]);
        assert_eq!(empty_owned.strides(), &[0, 3, 1]);
        assert!(empty_owned.data().is_empty());
    }
}
