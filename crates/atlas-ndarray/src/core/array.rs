use super::traits::Numeric;
use crate::{AtlasNdResult, internal::validate_owned_array_invariants, layout::compute_strides};

#[derive(Clone, Debug)]
pub struct NDArray<T: Numeric> {
    pub(crate) data: Vec<T>,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
}

impl<T: Numeric> NDArray<T> {
    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn is_contiguous(&self) -> bool {
        debug_assert_eq!(self.shape.len(), self.strides.len());
        self.strides == compute_strides(&self.shape)
    }

    pub(crate) fn validate_invariants(&self) -> AtlasNdResult<()> {
        validate_owned_array_invariants(self.data.len(), &self.shape, &self.strides)
    }
}

#[cfg(test)]
mod tests {
    use crate::NDArray;

    #[test]
    fn scalar_arrays_are_contiguous() {
        let array = NDArray::new(vec![], 7_i32);

        assert_eq!(array.len(), 1);
        assert_eq!(array.ndim(), 0);
        assert!(array.is_contiguous());
        assert_eq!(array.strides(), &[] as &[usize]);
        assert_eq!(array.validate_invariants(), Ok(()));
    }
}
