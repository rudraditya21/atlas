use crate::{
    ArrayElement, AtlasNdError, AtlasNdResult, ShapeArg, checked_compute_strides,
    checked_element_count, view::ArrayView,
};

impl<'a, T: ArrayElement> ArrayView<'a, T> {
    pub fn reshape<S>(self, new_shape: S) -> AtlasNdResult<Self>
    where
        S: ShapeArg,
    {
        let new_shape = new_shape.into_shape_vec();
        let old_size = checked_element_count(&self.shape)?;
        let new_size = checked_element_count(&new_shape)?;

        if old_size != new_size {
            return Err(AtlasNdError::InvalidReshape {
                from: self.shape.clone(),
                to: new_shape,
                reason: "element count must remain unchanged",
            });
        }

        if !can_reshape_view(&self.shape, &self.strides) {
            return Err(AtlasNdError::InvalidReshape {
                from: self.shape.clone(),
                to: new_shape,
                reason: "only contiguous or empty views can be reshaped",
            });
        }

        let new_strides = checked_compute_strides(&new_shape)?;
        ArrayView::from_parts(self.data, self.offset, new_shape, new_strides)
    }
}

fn can_reshape_view(shape: &[usize], strides: &[usize]) -> bool {
    crate::element_count(shape) == 0
        || crate::internal::layout::is_contiguous_layout(shape, strides)
}

#[cfg(test)]
mod tests {
    use crate::{AtlasNdError, NDArray};

    #[test]
    fn reshape_allows_contiguous_views() {
        let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let reshaped = array.view().reshape([3, 2]).unwrap();

        assert_eq!(reshaped.shape(), &[3, 2]);
        assert_eq!(reshaped.strides(), &[2, 1]);
        assert_eq!(*reshaped.get(&[2, 1]).unwrap(), 5);
    }

    #[test]
    fn reshape_rejects_non_contiguous_views() {
        let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let view = array.view().slice([0, 1], [2, 2]).unwrap();
        let error = view.reshape([4]).unwrap_err();

        assert_eq!(
            error,
            AtlasNdError::InvalidReshape {
                from: vec![2, 2],
                to: vec![4],
                reason: "only contiguous or empty views can be reshaped",
            }
        );
    }

    #[test]
    fn reshape_rejects_different_element_counts() {
        let array = NDArray::new(vec![2, 3], 0_i32).unwrap();
        let error = array.view().reshape([5]).unwrap_err();

        assert_eq!(
            error,
            AtlasNdError::InvalidReshape {
                from: vec![2, 3],
                to: vec![5],
                reason: "element count must remain unchanged",
            }
        );
    }

    #[test]
    fn reshape_supports_scalar_and_zero_length_contiguous_views() {
        let scalar = NDArray::new([], 9_i32).unwrap();
        let zero_length = NDArray::<i32>::zeros([2, 0, 3]).unwrap();

        let reshaped_scalar = scalar.view().reshape([1]).unwrap();
        let reshaped_zero_length = zero_length.view().reshape([0]).unwrap();

        assert_eq!(reshaped_scalar.shape(), &[1]);
        assert_eq!(reshaped_scalar.strides(), &[1]);
        assert_eq!(*reshaped_scalar.get(&[0]).unwrap(), 9);

        assert_eq!(reshaped_zero_length.shape(), &[0]);
        assert_eq!(reshaped_zero_length.strides(), &[1]);
        assert!(reshaped_zero_length.is_empty());
    }

    #[test]
    fn reshape_supports_zero_length_non_contiguous_views() {
        let array = NDArray::from_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let empty = array.view().slice([2, 3], [0, 0]).unwrap();
        let reshaped = empty.reshape([0]).unwrap();

        assert_eq!(reshaped.shape(), &[0]);
        assert_eq!(reshaped.strides(), &[1]);
        assert_eq!(reshaped.offset(), 0);
        assert!(reshaped.is_empty());
    }

    #[test]
    fn reshape_reports_scalar_element_count_errors_consistently() {
        let scalar = NDArray::new([], 1_i32).unwrap();

        assert_eq!(
            scalar.view().reshape([2]).unwrap_err(),
            AtlasNdError::InvalidReshape {
                from: vec![],
                to: vec![2],
                reason: "element count must remain unchanged",
            }
        );
    }

    #[test]
    fn reshape_reports_shape_overflow_explicitly() {
        let scalar = NDArray::new([], 1_i32).unwrap();

        assert_eq!(
            scalar.view().reshape([usize::MAX, 2]).unwrap_err(),
            AtlasNdError::ShapeOverflow { op: "element count", shape: vec![usize::MAX, 2] }
        );
    }
}
