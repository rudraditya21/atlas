use super::{
    array::NDArray,
    error::{AtlasNdError, AtlasNdResult},
    stride::{compute_strides, element_count},
    traits::Numeric,
};
use num_traits::Float;

impl<T: Numeric> NDArray<T> {
    /// Creates a dense row-major array filled with `value`.
    pub fn new<S>(shape: S, value: T) -> Self
    where
        S: AsRef<[usize]>,
    {
        Self::full(shape, value)
    }

    /// Creates a dense row-major array filled with `value`.
    pub fn full<S>(shape: S, value: T) -> Self
    where
        S: AsRef<[usize]>,
    {
        let shape = shape.as_ref().to_vec();
        let size = element_count(&shape);

        Self { data: vec![value; size], strides: compute_strides(&shape), shape }
    }

    /// Creates a dense row-major array filled with zeros.
    pub fn zeros<S>(shape: S) -> Self
    where
        S: AsRef<[usize]>,
    {
        Self::full(shape, T::zero())
    }

    /// Creates a dense row-major array filled with ones.
    pub fn ones<S>(shape: S) -> Self
    where
        S: AsRef<[usize]>,
    {
        Self::full(shape, T::one())
    }

    /// Creates a square identity matrix with ones on the main diagonal.
    pub fn eye(size: usize) -> Self {
        let mut data = vec![T::zero(); size * size];

        for index in 0..size {
            data[index * size + index] = T::one();
        }

        Self { data, strides: compute_strides(&[size, size]), shape: vec![size, size] }
    }

    /// Creates a dense row-major array from an explicit shape and backing data.
    pub fn from_shape_vec<S>(shape: S, data: Vec<T>) -> AtlasNdResult<Self>
    where
        S: AsRef<[usize]>,
    {
        let shape = shape.as_ref().to_vec();
        let expected = element_count(&shape);

        if expected != data.len() {
            return Err(AtlasNdError::ShapeMismatch { expected, actual: data.len() });
        }

        Ok(Self { data, strides: compute_strides(&shape), shape })
    }

    /// Creates a dense row-major array from an explicit shape and backing data.
    pub fn from_vec<S>(shape: S, data: Vec<T>) -> AtlasNdResult<Self>
    where
        S: AsRef<[usize]>,
    {
        Self::from_shape_vec(shape, data)
    }
}

impl<T> NDArray<T>
where
    T: Numeric + PartialOrd,
{
    /// Creates a 1D array from a half-open interval `[start, end)` with `step`.
    pub fn arange(start: T, end: T, step: T) -> AtlasNdResult<Self> {
        let zero = T::zero();

        if step == zero {
            return Err(AtlasNdError::InvalidArgument {
                op: "arange",
                reason: "step must be non-zero",
            });
        }

        if step > zero && start > end {
            return Err(AtlasNdError::InvalidArgument {
                op: "arange",
                reason: "positive step does not advance toward end",
            });
        }

        if step < zero && start < end {
            return Err(AtlasNdError::InvalidArgument {
                op: "arange",
                reason: "negative step does not advance toward end",
            });
        }

        let mut data = Vec::new();
        let mut current = start;

        if step > zero {
            while current < end {
                data.push(current);
                current += step;
            }
        } else {
            while current > end {
                data.push(current);
                current += step;
            }
        }

        Self::from_shape_vec(vec![data.len()], data)
    }
}

impl<T> NDArray<T>
where
    T: Numeric + Float,
{
    /// Creates a 1D array with `num` evenly spaced points from `start` to `end`, inclusive.
    pub fn linspace(start: T, end: T, num: usize) -> AtlasNdResult<Self> {
        if !start.is_finite() || !end.is_finite() {
            return Err(AtlasNdError::InvalidArgument {
                op: "linspace",
                reason: "start and end must be finite",
            });
        }

        if num == 0 {
            return Self::from_shape_vec(vec![0], Vec::new());
        }

        if num == 1 {
            return Self::from_shape_vec(vec![1], vec![start]);
        }

        let step = (end - start) / T::from(num - 1).expect("usize to float conversion");
        let mut data = Vec::with_capacity(num);

        for index in 0..num {
            if index == num - 1 {
                data.push(end);
            } else {
                data.push(start + step * T::from(index).expect("usize to float conversion"));
            }
        }

        Self::from_shape_vec(vec![num], data)
    }
}

#[cfg(test)]
mod tests {
    use crate::error::AtlasNdError;

    use super::NDArray;

    #[test]
    fn new_builds_a_contiguous_row_major_array() {
        let array = NDArray::new([2, 3], 5_i32);

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

        assert_eq!(error, AtlasNdError::ShapeMismatch { expected: 4, actual: 3 });
    }

    #[test]
    fn full_zeros_ones_and_from_shape_vec_provide_stable_constructor_surface() {
        let full = NDArray::full([2, 2], 9_i32);
        let zeros = NDArray::<i32>::zeros([2, 2]);
        let ones = NDArray::<i32>::ones([2, 2]);
        let from_shape_vec = NDArray::from_shape_vec([2, 2], vec![1_i32, 2, 3, 4]).unwrap();

        assert_eq!(full.data(), &[9, 9, 9, 9]);
        assert_eq!(zeros.data(), &[0, 0, 0, 0]);
        assert_eq!(ones.data(), &[1, 1, 1, 1]);
        assert_eq!(from_shape_vec.data(), &[1, 2, 3, 4]);
        assert!(full.is_contiguous());
        assert!(zeros.is_contiguous());
        assert!(ones.is_contiguous());
        assert!(from_shape_vec.is_contiguous());
    }

    #[test]
    fn eye_creates_a_contiguous_identity_matrix() {
        let identity = NDArray::<i32>::eye(3);

        assert_eq!(identity.shape(), &[3, 3]);
        assert_eq!(identity.strides(), &[3, 1]);
        assert_eq!(identity.data(), &[1, 0, 0, 0, 1, 0, 0, 0, 1]);
        assert!(identity.is_contiguous());
    }

    #[test]
    fn arange_supports_positive_and_negative_steps() {
        let forward = NDArray::arange(0_i32, 5, 2).unwrap();
        let backward = NDArray::arange(5_i32, 0, -2).unwrap();

        assert_eq!(forward.shape(), &[3]);
        assert_eq!(forward.data(), &[0, 2, 4]);
        assert_eq!(backward.shape(), &[3]);
        assert_eq!(backward.data(), &[5, 3, 1]);
        assert!(forward.is_contiguous());
        assert!(backward.is_contiguous());
    }

    #[test]
    fn arange_rejects_invalid_step_configuration() {
        assert_eq!(
            NDArray::arange(0_i32, 5, 0).unwrap_err(),
            AtlasNdError::InvalidArgument { op: "arange", reason: "step must be non-zero" }
        );

        assert_eq!(
            NDArray::arange(5_i32, 0, 1).unwrap_err(),
            AtlasNdError::InvalidArgument {
                op: "arange",
                reason: "positive step does not advance toward end",
            }
        );
    }

    #[test]
    fn linspace_creates_evenly_spaced_points() {
        let values = NDArray::linspace(0.0_f64, 1.0, 5).unwrap();
        let singleton = NDArray::linspace(2.5_f64, 9.0, 1).unwrap();
        let empty = NDArray::linspace(0.0_f64, 1.0, 0).unwrap();

        assert_eq!(values.shape(), &[5]);
        assert_eq!(values.data(), &[0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(singleton.data(), &[2.5]);
        assert_eq!(empty.shape(), &[0]);
        assert!(values.is_contiguous());
    }

    #[test]
    fn constructors_accept_slice_like_shape_arguments() {
        let dynamic_shape = vec![2, 3];

        let from_slice = NDArray::<i32>::zeros(dynamic_shape.as_slice());
        let from_array_ref = NDArray::<i32>::ones([2, 3]);

        assert_eq!(from_slice.shape(), &[2, 3]);
        assert_eq!(from_array_ref.shape(), &[2, 3]);
    }
}
