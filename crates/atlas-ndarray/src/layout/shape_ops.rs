use crate::{
    ArrayElement, AtlasNdError, AtlasNdResult, AxisIndex, NDArray,
    core::axis::{normalize_axis, normalize_insertion_axis},
    view::ArrayView,
};

impl<T: ArrayElement> NDArray<T> {
    pub fn squeeze(&self) -> ArrayView<'_, T> {
        self.view().squeeze()
    }

    pub fn squeeze_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<ArrayView<'_, T>> {
        self.view().squeeze_axis(axis)
    }

    pub fn expand_dims<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<ArrayView<'_, T>> {
        self.view().expand_dims(axis)
    }
}

impl<'a, T: ArrayElement> ArrayView<'a, T> {
    pub fn squeeze(self) -> ArrayView<'a, T> {
        let mut squeezed_shape = Vec::with_capacity(self.shape.len());
        let mut squeezed_strides = Vec::with_capacity(self.strides.len());

        for (&dim, &stride) in self.shape.iter().zip(self.strides.iter()) {
            if dim != 1 {
                squeezed_shape.push(dim);
                squeezed_strides.push(stride);
            }
        }

        ArrayView::from_parts(self.data, self.offset, squeezed_shape, squeezed_strides)
            .expect("squeezing a valid view must preserve valid metadata")
    }

    pub fn squeeze_axis<A: AxisIndex>(mut self, axis: A) -> AtlasNdResult<ArrayView<'a, T>> {
        let axis = normalize_axis(axis, self.shape.len())?;
        if self.shape[axis] != 1 {
            return Err(AtlasNdError::InvalidArgument {
                op: "squeeze",
                reason: "axis must have length 1",
            });
        }

        self.shape.remove(axis);
        self.strides.remove(axis);
        ArrayView::from_parts(self.data, self.offset, self.shape, self.strides)
    }

    pub fn expand_dims<A: AxisIndex>(mut self, axis: A) -> AtlasNdResult<ArrayView<'a, T>> {
        let axis = normalize_insertion_axis(axis, self.shape.len())?;
        let inserted_stride = if axis == self.shape.len() {
            1
        } else {
            self.strides[axis].checked_mul(self.shape[axis]).ok_or_else(|| {
                AtlasNdError::ShapeOverflow { op: "expand_dims", shape: self.shape.clone() }
            })?
        };

        self.shape.insert(axis, 1);
        self.strides.insert(axis, inserted_stride);
        ArrayView::from_parts(self.data, self.offset, self.shape, self.strides)
    }
}

#[cfg(test)]
mod tests {
    use crate::{AtlasNdError, NDArray};

    #[test]
    fn squeeze_removes_all_singleton_axes_without_copying() {
        let array = NDArray::from_shape_vec([1, 2, 1, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let squeezed = array.squeeze();

        assert_eq!(squeezed.shape(), &[2, 3]);
        assert_eq!(squeezed.strides(), &[3, 1]);
        assert_eq!(squeezed.data(), array.data());
        assert!(squeezed.is_contiguous());
    }

    #[test]
    fn squeeze_preserves_scalar_and_non_singleton_shapes() {
        let scalar = NDArray::from_shape_vec([], vec![7_i32]).unwrap();
        let vector = NDArray::from_shape_vec([3], vec![1_i32, 2, 3]).unwrap();

        assert_eq!(scalar.squeeze().shape(), &[] as &[usize]);
        assert_eq!(scalar.squeeze().strides(), &[] as &[usize]);
        assert_eq!(vector.squeeze().shape(), &[3]);
        assert_eq!(vector.squeeze().strides(), &[1]);
    }

    #[test]
    fn squeeze_axis_removes_a_single_valid_singleton_axis() {
        let array = NDArray::from_shape_vec([2, 1, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let squeezed = array.squeeze_axis(1).unwrap();

        assert_eq!(squeezed.shape(), &[2, 3]);
        assert_eq!(squeezed.strides(), &[3, 1]);
        assert_eq!(squeezed.data(), array.data());
    }

    #[test]
    fn squeeze_axis_supports_negative_axes() {
        let array = NDArray::from_shape_vec([2, 1, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let squeezed = array.squeeze_axis(-2).unwrap();

        assert_eq!(squeezed.shape(), &[2, 3]);
        assert_eq!(squeezed.strides(), &[3, 1]);
        assert_eq!(squeezed.data(), array.data());
    }

    #[test]
    fn squeeze_axis_rejects_non_singleton_and_invalid_axes() {
        let array = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();

        assert_eq!(
            array.squeeze_axis(1).unwrap_err(),
            AtlasNdError::InvalidArgument { op: "squeeze", reason: "axis must have length 1" }
        );
        assert_eq!(
            array.squeeze_axis(2).unwrap_err(),
            AtlasNdError::InvalidAxis { axis: 2, ndim: 2 }
        );
    }

    #[test]
    fn squeeze_and_expand_dims_preserve_strided_view_metadata() {
        let array = NDArray::from_shape_vec([2, 1, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let view = array.view().transpose();
        let squeezed = view.clone().squeeze_axis(1).unwrap();
        let expanded = squeezed.clone().expand_dims(1).unwrap();

        assert_eq!(view.shape(), &[3, 1, 2]);
        assert_eq!(view.strides(), &[1, 3, 3]);
        assert_eq!(squeezed.shape(), &[3, 2]);
        assert_eq!(squeezed.strides(), &[1, 3]);
        assert_eq!(expanded.shape(), &[3, 1, 2]);
        assert_eq!(expanded.strides(), &[1, 6, 3]);
        assert_eq!(expanded.data(), array.data());
    }

    #[test]
    fn expand_dims_inserts_singleton_axes_at_valid_positions() {
        let array = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();

        assert_eq!(array.expand_dims(0).unwrap().shape(), &[1, 2, 3]);
        assert_eq!(array.expand_dims(1).unwrap().shape(), &[2, 1, 3]);
        assert_eq!(array.expand_dims(2).unwrap().shape(), &[2, 3, 1]);
        assert_eq!(array.expand_dims(-1).unwrap().shape(), &[2, 3, 1]);
        assert_eq!(array.expand_dims(-3).unwrap().shape(), &[1, 2, 3]);
    }

    #[test]
    fn expand_dims_assigns_predictable_strides_for_contiguous_and_empty_inputs() {
        let contiguous = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let empty = NDArray::<i32>::from_shape_vec([2, 0, 3], Vec::new()).unwrap();
        let scalar = NDArray::from_shape_vec([], vec![7_i32]).unwrap();

        assert_eq!(contiguous.expand_dims(0).unwrap().strides(), &[6, 3, 1]);
        assert_eq!(contiguous.expand_dims(1).unwrap().strides(), &[3, 3, 1]);
        assert_eq!(contiguous.expand_dims(2).unwrap().strides(), &[3, 1, 1]);
        assert_eq!(empty.expand_dims(1).unwrap().strides(), &[0, 0, 3, 1]);
        assert_eq!(scalar.expand_dims(0).unwrap().shape(), &[1]);
        assert_eq!(scalar.expand_dims(0).unwrap().strides(), &[1]);
    }

    #[test]
    fn expand_dims_rejects_invalid_axes() {
        let array = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();

        assert_eq!(
            array.expand_dims(3).unwrap_err(),
            AtlasNdError::InvalidAxis { axis: 3, ndim: 2 }
        );
        assert_eq!(
            array.expand_dims(-4).unwrap_err(),
            AtlasNdError::InvalidAxis { axis: -4, ndim: 2 }
        );
    }

    #[test]
    fn expand_dims_supports_negative_axes_on_views() {
        let array = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let view = array.view().transpose();
        let expanded = view.expand_dims(-1).unwrap();

        assert_eq!(expanded.shape(), &[3, 2, 1]);
        assert_eq!(expanded.strides(), &[1, 3, 1]);
        assert_eq!(*expanded.get(&[-1, -1, 0]).unwrap(), 5);
    }
}
