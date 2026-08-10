use crate::{AsArray, NDArray, Numeric, view::ArrayView};

impl<T: Numeric> NDArray<T> {
    pub fn ravel(&self) -> AsArray<'_, T> {
        self.view().ravel()
    }
}

impl<'a, T: Numeric> ArrayView<'a, T> {
    pub fn ravel(&self) -> AsArray<'a, T> {
        if self.is_empty() {
            return AsArray::Borrowed(ArrayView {
                data: self.data,
                offset: self.offset,
                shape: vec![0],
                strides: vec![1],
            });
        }

        if self.is_contiguous() {
            return AsArray::Borrowed(
                self.clone()
                    .reshape([self.len()])
                    .expect("contiguous views can always be flattened without copying"),
            );
        }

        let data = crate::internal::value_iter(self.data, self.offset, &self.shape, &self.strides)
            .copied()
            .collect();

        AsArray::Owned(
            NDArray::from_vector_data(data)
                .expect("valid views can always be flattened into owned contiguous arrays"),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::{AsArray, NDArray};

    #[test]
    fn ravel_preserves_contiguous_owned_arrays_as_borrowed_views() {
        let array = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let raveled = array.ravel();

        assert!(raveled.is_borrowed());
        assert_eq!(raveled.view().shape(), &[6]);
        assert_eq!(raveled.view().strides(), &[1]);
        assert_eq!(raveled.view().data(), array.data());
    }

    #[test]
    fn ravel_materializes_non_contiguous_views_in_logical_order() {
        let array = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let raveled = array.view().transpose().ravel();

        assert!(raveled.is_owned());
        assert_eq!(raveled.view().shape(), &[6]);
        assert_eq!(raveled.view().strides(), &[1]);
        assert_eq!(raveled.into_owned().data(), &[0, 3, 1, 4, 2, 5]);
    }

    #[test]
    fn ravel_preserves_scalar_and_zero_length_views_without_copying() {
        let scalar = NDArray::from_shape_vec([], vec![7_i32]).unwrap();
        let empty = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let scalar_raveled = scalar.ravel();
        let empty_raveled = empty.view().slice([2, 3], [0, 0]).unwrap().ravel();

        assert!(scalar_raveled.is_borrowed());
        assert_eq!(scalar_raveled.view().shape(), &[1]);
        assert_eq!(scalar_raveled.view().strides(), &[1]);
        assert_eq!(scalar_raveled.view().data(), &[7]);

        assert!(empty_raveled.is_borrowed());
        assert_eq!(empty_raveled.view().shape(), &[0]);
        assert_eq!(empty_raveled.view().strides(), &[1]);
        assert_eq!(empty_raveled.view().len(), 0);
        assert_eq!(empty_raveled.view().dense_slice(), Some(&[] as &[i32]));
    }

    #[test]
    fn ravel_keeps_vector_inputs_as_borrowed_1d_views() {
        let vector = NDArray::from_shape_vec([3], vec![1_i32, 2, 3]).unwrap();
        let raveled = vector.ravel();

        assert!(matches!(&raveled, AsArray::Borrowed(_)));
        assert_eq!(raveled.view().shape(), &[3]);
        assert_eq!(raveled.view().strides(), &[1]);
        assert_eq!(raveled.view().data(), &[1, 2, 3]);
    }
}
