use super::{
    array::NDArray,
    error::{AtlasNdError, AtlasNdResult},
    stride::{compute_strides, element_count},
    traits::Numeric,
};

impl<T: Numeric> NDArray<T> {
    /// Creates a dense row-major array filled with `value`.
    pub fn new(shape: Vec<usize>, value: T) -> Self {
        Self::full(shape, value)
    }

    /// Creates a dense row-major array filled with `value`.
    pub fn full(shape: Vec<usize>, value: T) -> Self {
        let size = element_count(&shape);

        Self {
            data: vec![value; size],
            strides: compute_strides(&shape),
            shape,
        }
    }

    /// Creates a dense row-major array filled with zeros.
    pub fn zeros(shape: Vec<usize>) -> Self {
        Self::full(shape, T::zero())
    }

    /// Creates a dense row-major array filled with ones.
    pub fn ones(shape: Vec<usize>) -> Self {
        Self::full(shape, T::one())
    }

    /// Creates a dense row-major array from an explicit shape and backing data.
    pub fn from_shape_vec(shape: Vec<usize>, data: Vec<T>) -> AtlasNdResult<Self> {
        let expected = element_count(&shape);

        if expected != data.len() {
            return Err(AtlasNdError::ShapeMismatch {
                expected,
                actual: data.len(),
            });
        }

        Ok(Self {
            data,
            strides: compute_strides(&shape),
            shape,
        })
    }

    /// Creates a dense row-major array from an explicit shape and backing data.
    pub fn from_vec(shape: Vec<usize>, data: Vec<T>) -> AtlasNdResult<Self> {
        Self::from_shape_vec(shape, data)
    }
}

#[cfg(test)]
mod tests {
    use crate::error::AtlasNdError;

    use super::NDArray;

    #[test]
    fn new_builds_a_contiguous_row_major_array() {
        let array = NDArray::new(vec![2, 3], 5_i32);

        assert_eq!(array.len(), 6);
        assert_eq!(array.ndim(), 2);
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.strides(), &[3, 1]);
        assert!(array.is_contiguous());
        assert_eq!(array.data(), &[5, 5, 5, 5, 5, 5]);
    }

    #[test]
    fn from_vec_preserves_data_for_contiguous_layout() {
        let array = NDArray::from_vec(vec![2, 2], vec![1_i32, 2, 3, 4]).unwrap();

        assert_eq!(array.shape(), &[2, 2]);
        assert_eq!(array.strides(), &[2, 1]);
        assert!(array.is_contiguous());
        assert_eq!(array.data(), &[1, 2, 3, 4]);
    }

    #[test]
    fn from_vec_rejects_inconsistent_shape_and_data_length() {
        let error = NDArray::from_vec(vec![2, 2], vec![1_i32, 2, 3]).unwrap_err();

        assert_eq!(
            error,
            AtlasNdError::ShapeMismatch {
                expected: 4,
                actual: 3
            }
        );
    }

    #[test]
    fn full_zeros_ones_and_from_shape_vec_provide_stable_constructor_surface() {
        let full = NDArray::full(vec![2, 2], 9_i32);
        let zeros = NDArray::<i32>::zeros(vec![2, 2]);
        let ones = NDArray::<i32>::ones(vec![2, 2]);
        let from_shape_vec = NDArray::from_shape_vec(vec![2, 2], vec![1_i32, 2, 3, 4]).unwrap();

        assert_eq!(full.data(), &[9, 9, 9, 9]);
        assert_eq!(zeros.data(), &[0, 0, 0, 0]);
        assert_eq!(ones.data(), &[1, 1, 1, 1]);
        assert_eq!(from_shape_vec.data(), &[1, 2, 3, 4]);
        assert!(full.is_contiguous());
        assert!(zeros.is_contiguous());
        assert!(ones.is_contiguous());
        assert!(from_shape_vec.is_contiguous());
    }
}
