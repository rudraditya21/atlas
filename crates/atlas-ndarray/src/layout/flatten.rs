use crate::{NDArray, Numeric, view::ArrayView};

impl<T: Numeric> NDArray<T> {
    pub fn flatten(&self) -> NDArray<T> {
        self.ravel().into_owned()
    }
}

impl<'a, T: Numeric> ArrayView<'a, T> {
    pub fn flatten(&self) -> NDArray<T> {
        self.ravel().into_owned()
    }
}

#[cfg(test)]
mod tests {
    use crate::NDArray;

    #[test]
    fn flatten_copies_contiguous_arrays_into_owned_vectors() {
        let mut array = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let flattened = array.flatten();

        *array.get_mut(&[0, 0]).unwrap() = 99;

        assert_eq!(flattened.shape(), &[6]);
        assert_eq!(flattened.strides(), &[1]);
        assert_eq!(flattened.data(), &[0, 1, 2, 3, 4, 5]);
        assert!(flattened.is_contiguous());
    }

    #[test]
    fn flatten_copies_strided_views_in_logical_order() {
        let array = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let flattened = array.view().transpose().flatten();

        assert_eq!(flattened.shape(), &[6]);
        assert_eq!(flattened.strides(), &[1]);
        assert_eq!(flattened.data(), &[0, 3, 1, 4, 2, 5]);
        assert!(flattened.is_contiguous());
    }

    #[test]
    fn flatten_normalizes_scalar_and_zero_length_inputs_to_owned_1d_arrays() {
        let scalar = NDArray::from_shape_vec([], vec![7_i32]).unwrap();
        let empty_source = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let empty = empty_source.view().slice([2, 3], [0, 0]).unwrap();

        let scalar_flattened = scalar.flatten();
        let empty_flattened = empty.flatten();

        assert_eq!(scalar_flattened.shape(), &[1]);
        assert_eq!(scalar_flattened.strides(), &[1]);
        assert_eq!(scalar_flattened.data(), &[7]);

        assert_eq!(empty_flattened.shape(), &[0]);
        assert_eq!(empty_flattened.strides(), &[1]);
        assert!(empty_flattened.data().is_empty());
    }
}
