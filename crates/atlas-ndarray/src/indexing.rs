use super::{
    array::NDArray,
    error::{AtlasNdError, AtlasNdResult},
    traits::Numeric,
};

impl<T: Numeric> NDArray<T> {
    fn offset(&self, indices: &[usize]) -> AtlasNdResult<usize> {
        debug_assert_eq!(self.shape.len(), self.strides.len());
        if indices.len() != self.shape.len() {
            return Err(AtlasNdError::DimensionMismatch {
                expected: self.shape.len(),
                actual: indices.len(),
            });
        }

        let mut offset = 0;
        for (axis, ((index, dim), stride)) in
            indices.iter().zip(self.shape.iter()).zip(self.strides.iter()).enumerate()
        {
            if *index >= *dim {
                return Err(AtlasNdError::IndexOutOfBounds { axis, index: *index, dim: *dim });
            }
            offset += index * stride;
        }

        Ok(offset)
    }

    pub fn get(&self, indices: &[usize]) -> AtlasNdResult<&T> {
        let idx = self.offset(indices)?;
        Ok(&self.data[idx])
    }

    pub fn get_mut(&mut self, indices: &[usize]) -> AtlasNdResult<&mut T> {
        let idx = self.offset(indices)?;
        Ok(&mut self.data[idx])
    }
}

#[cfg(test)]
mod tests {
    use crate::error::AtlasNdError;

    use super::NDArray;

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
        let array = NDArray::new(vec![2, 3], 0_i32);

        let error = array.get(&[0]).unwrap_err();

        assert_eq!(error, AtlasNdError::DimensionMismatch { expected: 2, actual: 1 });
    }

    #[test]
    fn get_rejects_out_of_bounds_indices() {
        let array = NDArray::new(vec![2, 3], 0_i32);

        let error = array.get(&[2, 0]).unwrap_err();

        assert_eq!(error, AtlasNdError::IndexOutOfBounds { axis: 0, index: 2, dim: 2 });
    }

    #[test]
    fn get_supports_scalar_arrays_and_rejects_scalar_index_mismatch() {
        let array = NDArray::new([], 7_i32);

        assert_eq!(*array.get(&[]).unwrap(), 7);
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
        let mut array = NDArray::new([2, 2], 0_i32);

        assert_eq!(
            array.get_mut(&[0]).unwrap_err(),
            AtlasNdError::DimensionMismatch { expected: 2, actual: 1 }
        );
        assert_eq!(
            array.get_mut(&[0, 2]).unwrap_err(),
            AtlasNdError::IndexOutOfBounds { axis: 1, index: 2, dim: 2 }
        );
    }
}
