use crate::{AtlasNdError, AtlasNdResult, NDArray, Numeric};
use num_traits::Float;

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
    use crate::{AtlasNdError, NDArray};

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
    fn arange_handles_empty_equal_endpoint_ranges() {
        let increasing = NDArray::arange(3_i32, 3, 1).unwrap();
        let decreasing = NDArray::arange(3_i32, 3, -1).unwrap();

        assert_eq!(increasing.shape(), &[0]);
        assert!(increasing.data().is_empty());
        assert_eq!(decreasing.shape(), &[0]);
        assert!(decreasing.data().is_empty());
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

        assert_eq!(
            NDArray::arange(0_i32, 5, -1).unwrap_err(),
            AtlasNdError::InvalidArgument {
                op: "arange",
                reason: "negative step does not advance toward end",
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
    fn linspace_handles_descending_and_degenerate_ranges() {
        let descending = NDArray::linspace(3.0_f64, -1.0, 3).unwrap();
        let degenerate = NDArray::linspace(4.5_f64, 4.5, 4).unwrap();

        assert_eq!(descending.data(), &[3.0, 1.0, -1.0]);
        assert_eq!(degenerate.data(), &[4.5, 4.5, 4.5, 4.5]);
        assert!(descending.is_contiguous());
        assert!(degenerate.is_contiguous());
    }

    #[test]
    fn linspace_rejects_non_finite_endpoints() {
        assert_eq!(
            NDArray::linspace(f64::NAN, 1.0, 3).unwrap_err(),
            AtlasNdError::InvalidArgument {
                op: "linspace",
                reason: "start and end must be finite",
            }
        );

        assert_eq!(
            NDArray::linspace(0.0_f64, f64::INFINITY, 3).unwrap_err(),
            AtlasNdError::InvalidArgument {
                op: "linspace",
                reason: "start and end must be finite",
            }
        );
    }
}
