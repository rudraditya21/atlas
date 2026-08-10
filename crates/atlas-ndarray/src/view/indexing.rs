use crate::{AtlasNdResult, AxisIndex, NDArray, Numeric, core::axis::normalize_and_offset_indices};

impl<T: Numeric> NDArray<T> {
    fn offset<I: AxisIndex>(&self, indices: &[I]) -> AtlasNdResult<usize> {
        normalize_and_offset_indices(0, indices, &self.shape, &self.strides)
    }

    pub fn get<I: AxisIndex>(&self, indices: &[I]) -> AtlasNdResult<&T> {
        let idx = self.offset(indices)?;
        Ok(&self.data[idx])
    }

    pub fn get_mut<I: AxisIndex>(&mut self, indices: &[I]) -> AtlasNdResult<&mut T> {
        let idx = self.offset(indices)?;
        Ok(&mut self.data[idx])
    }
}

#[cfg(test)]
mod tests {
    use crate::{AtlasNdError, NDArray};

    #[test]
    fn get_uses_row_major_offsets_for_two_dimensional_arrays() {
        let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();

        assert_eq!(*array.get(&[0, 0]).unwrap(), 0);
        assert_eq!(*array.get(&[0, 1]).unwrap(), 1);
        assert_eq!(*array.get(&[1, 0]).unwrap(), 3);
        assert_eq!(*array.get(&[1, 2]).unwrap(), 5);
    }

    #[test]
    fn get_uses_row_major_offsets_for_three_dimensional_arrays() {
        let array = NDArray::from_vec(vec![2, 3, 4], (0_i32..24).collect()).unwrap();

        assert_eq!(*array.get(&[0, 0, 0]).unwrap(), 0);
        assert_eq!(*array.get(&[0, 1, 2]).unwrap(), 6);
        assert_eq!(*array.get(&[1, 0, 0]).unwrap(), 12);
        assert_eq!(*array.get(&[1, 2, 3]).unwrap(), 23);
    }

    #[test]
    fn get_rejects_wrong_dimension_count() {
        let array = NDArray::new(vec![2, 3], 0_i32).unwrap();

        let error = array.get(&[0]).unwrap_err();

        assert_eq!(error, AtlasNdError::DimensionMismatch { expected: 2, actual: 1 });
    }

    #[test]
    fn get_rejects_out_of_bounds_indices() {
        let array = NDArray::new(vec![2, 3], 0_i32).unwrap();

        let error = array.get(&[2, 0]).unwrap_err();

        assert_eq!(error, AtlasNdError::IndexOutOfBounds { axis: 0, index: 2, dim: 2 });
    }

    #[test]
    fn get_supports_negative_indices_through_shared_normalization() {
        let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();

        assert_eq!(*array.get(&[-1, -1]).unwrap(), 5);
        assert_eq!(*array.get(&[-2, 1]).unwrap(), 1);
        assert_eq!(
            array.get(&[-3, 0]).unwrap_err(),
            AtlasNdError::IndexOutOfBounds { axis: 0, index: -3, dim: 2 }
        );
    }

    #[test]
    fn get_supports_scalar_arrays_and_rejects_scalar_index_mismatch() {
        let array = NDArray::new([], 7_i32).unwrap();

        assert_eq!(*array.get(&[] as &[i64]).unwrap(), 7);
        assert_eq!(
            array.get(&[0]).unwrap_err(),
            AtlasNdError::DimensionMismatch { expected: 0, actual: 1 }
        );
    }

    #[test]
    fn get_mut_updates_the_underlying_element() {
        let mut array = NDArray::from_vec(vec![2, 2], vec![1_i32, 2, 3, 4]).unwrap();

        *array.get_mut(&[1, 0]).unwrap() = 9;

        assert_eq!(array.data(), &[1, 2, 9, 4]);
    }

    #[test]
    fn get_mut_returns_consistent_dimension_and_bounds_errors() {
        let mut array = NDArray::new([2, 2], 0_i32).unwrap();

        assert_eq!(
            array.get_mut(&[0]).unwrap_err(),
            AtlasNdError::DimensionMismatch { expected: 2, actual: 1 }
        );
        assert_eq!(
            array.get_mut(&[0, 2]).unwrap_err(),
            AtlasNdError::IndexOutOfBounds { axis: 1, index: 2, dim: 2 }
        );
    }

    #[test]
    fn get_mut_supports_negative_indices() {
        let mut array = NDArray::from_vec(vec![2, 2], vec![1_i32, 2, 3, 4]).unwrap();

        *array.get_mut(&[-1, -2]).unwrap() = 9;

        assert_eq!(array.data(), &[1, 2, 9, 4]);
    }
}
