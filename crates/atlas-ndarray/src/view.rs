use super::{
    array::NDArray,
    error::{AtlasNdError, AtlasNdResult},
    stride::{compute_strides, element_count},
    traits::Numeric,
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
        ArrayView {
            data: &self.data,
            offset: 0,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
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
        debug_assert_eq!(self.shape.len(), self.strides.len());
        self.strides == compute_strides(&self.shape)
    }
}

#[cfg(test)]
mod tests {
    use crate::error::AtlasNdError;

    use super::NDArray;

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
}
