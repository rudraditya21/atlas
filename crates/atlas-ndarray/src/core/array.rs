use super::traits::Numeric;
use crate::{
    AtlasNdResult,
    internal::{
        layout::{dense_storage_slice, is_contiguous_layout},
        validate_owned_array_invariants,
    },
};

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
        is_contiguous_layout(&self.shape, &self.strides)
    }

    pub fn dense_slice(&self) -> &[T] {
        dense_storage_slice(&self.data, 0, &self.shape, &self.strides)
            .expect("owned arrays always expose a dense storage region")
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
        let array = NDArray::new(vec![], 7_i32).unwrap();

        assert_eq!(array.len(), 1);
        assert_eq!(array.ndim(), 0);
        assert!(array.is_contiguous());
        assert_eq!(array.strides(), &[] as &[usize]);
        assert_eq!(array.dense_slice(), &[7]);
        assert_eq!(array.validate_invariants(), Ok(()));
    }

    #[test]
    fn dense_slice_preserves_owned_storage_for_empty_and_multidimensional_arrays() {
        let matrix = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let empty = NDArray::<i32>::zeros([2, 0, 3]).unwrap();

        assert_eq!(matrix.dense_slice(), matrix.data());
        assert_eq!(empty.dense_slice(), &[] as &[i32]);
    }
}
