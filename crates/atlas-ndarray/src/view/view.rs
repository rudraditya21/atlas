use crate::{
    AtlasNdError, AtlasNdResult, NDArray, Numeric,
    internal::{
        layout::{dense_storage_slice, is_contiguous_layout},
        shape::element_count,
        validate_view_invariants,
    },
};

#[derive(Debug, Clone)]
pub struct ArrayView<'a, T: Numeric> {
    pub(crate) data: &'a [T],
    pub(crate) offset: usize,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
}

impl<T: Numeric> NDArray<T> {
    pub fn view(&self) -> ArrayView<'_, T> {
        self.validate_invariants()
            .unwrap_or_else(|error| panic!("NDArray::view failed invariant validation: {error}"));

        let view = ArrayView {
            data: &self.data,
            offset: 0,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        };

        view.validate_invariants()
            .unwrap_or_else(|error| panic!("NDArray::view failed invariant validation: {error}"));

        view
    }
}

impl<'a, T: Numeric> ArrayView<'a, T> {
    fn offset_for_index(&self, index: &[usize]) -> AtlasNdResult<usize> {
        debug_assert_eq!(self.shape.len(), self.strides.len());
        if index.len() != self.shape.len() {
            return Err(AtlasNdError::DimensionMismatch {
                expected: self.shape.len(),
                actual: index.len(),
            });
        }

        let mut offset = self.offset;

        for (axis, ((index, dim), stride)) in
            index.iter().zip(self.shape.iter()).zip(self.strides.iter()).enumerate()
        {
            if *index >= *dim {
                return Err(AtlasNdError::IndexOutOfBounds { axis, index: *index, dim: *dim });
            }
            offset += index * stride;
        }

        Ok(offset)
    }

    pub fn get(&self, index: &[usize]) -> AtlasNdResult<&T> {
        let idx = self.offset_for_index(index)?;
        Ok(&self.data[idx])
    }

    pub fn data(&self) -> &'a [T] {
        self.data
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn len(&self) -> usize {
        element_count(&self.shape)
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_contiguous(&self) -> bool {
        is_contiguous_layout(&self.shape, &self.strides)
    }

    pub fn dense_slice(&self) -> Option<&'a [T]> {
        dense_storage_slice(self.data, self.offset, &self.shape, &self.strides)
    }

    pub(crate) fn validate_invariants(&self) -> AtlasNdResult<()> {
        validate_view_invariants(self.data.len(), self.offset, &self.shape, &self.strides)
    }
}

#[cfg(test)]
mod tests {
    use crate::{AtlasNdError, NDArray};

    #[test]
    fn view_preserves_owned_layout_metadata() {
        let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let view = array.view();

        assert_eq!(view.data(), array.data());
        assert_eq!(view.offset(), 0);
        assert_eq!(view.shape(), &[2, 3]);
        assert_eq!(view.strides(), &[3, 1]);
        assert_eq!(view.len(), 6);
        assert_eq!(view.ndim(), 2);
        assert!(view.is_contiguous());
        assert!(view.dense_slice().is_some());
        assert_eq!(view.dense_slice().unwrap(), array.data());
        assert_eq!(view.validate_invariants(), Ok(()));
    }

    #[test]
    fn view_get_uses_underlying_array_storage() {
        let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let view = array.view();

        assert_eq!(*view.get(&[1, 2]).unwrap(), 5);
    }

    #[test]
    fn view_get_rejects_dimension_mismatch() {
        let array = NDArray::new(vec![2, 3], 0_i32);
        let view = array.view();

        let error = view.get(&[0]).unwrap_err();

        assert_eq!(error, AtlasNdError::DimensionMismatch { expected: 2, actual: 1 });
    }

    #[test]
    fn scalar_view_uses_empty_index_and_reports_rank_mismatch_consistently() {
        let array = NDArray::new([], 13_i32);
        let view = array.view();

        assert_eq!(*view.get(&[]).unwrap(), 13);
        assert_eq!(
            view.get(&[0]).unwrap_err(),
            AtlasNdError::DimensionMismatch { expected: 0, actual: 1 }
        );
    }

    #[test]
    fn view_get_reports_axis_specific_bounds_errors() {
        let array = NDArray::new([2, 3], 0_i32);
        let view = array.view();

        assert_eq!(
            view.get(&[0, 3]).unwrap_err(),
            AtlasNdError::IndexOutOfBounds { axis: 1, index: 3, dim: 3 }
        );
    }

    #[test]
    fn transposed_view_is_storage_dense_without_being_row_major_contiguous() {
        let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let view = array.view().transpose();

        assert!(!view.is_contiguous());
        assert!(view.dense_slice().is_some());
        assert_eq!(view.dense_slice().unwrap(), array.data());
        assert_eq!(view.validate_invariants(), Ok(()));
    }

    #[test]
    fn sliced_view_is_not_storage_dense_when_it_skips_elements() {
        let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let view = array.view().slice([0, 1], [2, 2]).unwrap();

        assert!(view.dense_slice().is_none());
    }

    #[test]
    fn empty_dense_views_return_valid_empty_slices() {
        let array = NDArray::from_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let view = array.view().slice([1, 3], [1, 0]).unwrap();

        assert!(view.dense_slice().is_some());
        assert_eq!(view.dense_slice().unwrap(), &[] as &[i32]);
        assert_eq!(view.validate_invariants(), Ok(()));
    }
}
