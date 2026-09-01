use crate::{AtlasNdError, AtlasNdResult, NDArray, Numeric};
use num_traits::Float;

#[doc(hidden)]
pub trait ArangeElement: Numeric + PartialOrd {
    fn validate_arange_inputs(start: Self, end: Self, step: Self) -> AtlasNdResult<()>;
    fn advance(current: Self, step: Self) -> Option<Self>;
}

macro_rules! impl_integer_arange_element {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl ArangeElement for $ty {
                fn validate_arange_inputs(_start: Self, _end: Self, _step: Self) -> AtlasNdResult<()> {
                    Ok(())
                }

                fn advance(current: Self, step: Self) -> Option<Self> {
                    current.checked_add(step)
                }
            }
        )+
    };
}

macro_rules! impl_float_arange_element {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl ArangeElement for $ty {
                fn validate_arange_inputs(start: Self, end: Self, step: Self) -> AtlasNdResult<()> {
                    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
                        return Err(AtlasNdError::InvalidArgument {
                            op: "arange",
                            reason: "start end and step must be finite",
                        });
                    }

                    Ok(())
                }

                fn advance(current: Self, step: Self) -> Option<Self> {
                    Some(current + step)
                }
            }
        )+
    };
}

impl_integer_arange_element!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);
impl_float_arange_element!(f32, f64);

impl<T> NDArray<T>
where
    T: ArangeElement,
{
    /// Creates a contiguous 1D array from the half-open interval `[start, end)` with `step`.
    ///
    /// Positive steps require `start < end`; negative steps require `start > end`; otherwise the
    /// result is empty. Integer ranges return an error when a step would overflow the dtype.
    /// Floating-point inputs must be finite and return an error when a step cannot advance.
    pub fn arange(start: T, end: T, step: T) -> AtlasNdResult<Self> {
        T::validate_arange_inputs(start, end, step)?;
        let zero = T::zero();

        if step == zero {
            return Err(AtlasNdError::InvalidArgument {
                op: "arange",
                reason: "step must be non-zero",
            });
        }

        if start == end || (step > zero && start > end) || (step < zero && start < end) {
            return Self::from_vector_data(Vec::new());
        }

        let mut data = Vec::new();
        let mut current = start;

        if step > zero {
            while current < end {
                data.push(current);
                let next = T::advance(current, step).ok_or(AtlasNdError::InvalidArgument {
                    op: "arange",
                    reason: "range overflows dtype",
                })?;
                if next <= current {
                    return Err(AtlasNdError::InvalidArgument {
                        op: "arange",
                        reason: "step must advance values",
                    });
                }
                current = next;
            }
        } else {
            while current > end {
                data.push(current);
                let next = T::advance(current, step).ok_or(AtlasNdError::InvalidArgument {
                    op: "arange",
                    reason: "range overflows dtype",
                })?;
                if next >= current {
                    return Err(AtlasNdError::InvalidArgument {
                        op: "arange",
                        reason: "step must advance values",
                    });
                }
                current = next;
            }
        }

        Self::from_vector_data(data)
    }
}

impl<T> NDArray<T>
where
    T: Numeric + Float,
{
    /// Creates a contiguous 1D float array with `num` evenly spaced points from `start` to `end`.
    ///
    /// Both endpoints are included when `num >= 2`; zero points produce an empty array and one
    /// point produces `[start]`. Endpoints and all generated values must remain finite.
    pub fn linspace(start: T, end: T, num: usize) -> AtlasNdResult<Self> {
        if !start.is_finite() || !end.is_finite() {
            return Err(AtlasNdError::InvalidArgument {
                op: "linspace",
                reason: "start and end must be finite",
            });
        }

        if num == 0 {
            return Self::from_vector_data(Vec::new());
        }

        if num == 1 {
            return Self::from_vector_data(vec![start]);
        }

        let step = (end - start) / T::from(num - 1).expect("usize to float conversion");
        if !step.is_finite() {
            return Err(AtlasNdError::InvalidArgument {
                op: "linspace",
                reason: "generated step must be finite",
            });
        }

        let mut data = Vec::with_capacity(num);

        for index in 0..num {
            if index == 0 {
                data.push(start);
            } else if index == num - 1 {
                data.push(end);
            } else {
                let value = start + step * T::from(index).expect("usize to float conversion");
                if !value.is_finite() {
                    return Err(AtlasNdError::InvalidArgument {
                        op: "linspace",
                        reason: "generated values must be finite",
                    });
                }
                data.push(value);
            }
        }

        Self::from_vector_data(data)
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
    fn arange_handles_empty_equal_and_direction_mismatched_ranges() {
        let increasing = NDArray::arange(3_i32, 3, 1).unwrap();
        let decreasing = NDArray::arange(3_i32, 3, -1).unwrap();
        let positive_mismatch = NDArray::arange(5_i32, 0, 1).unwrap();
        let negative_mismatch = NDArray::arange(0_i32, 5, -1).unwrap();

        assert_eq!(increasing.shape(), &[0]);
        assert!(increasing.data().is_empty());
        assert_eq!(decreasing.shape(), &[0]);
        assert!(decreasing.data().is_empty());
        assert_eq!(positive_mismatch.shape(), &[0]);
        assert!(positive_mismatch.data().is_empty());
        assert_eq!(negative_mismatch.shape(), &[0]);
        assert!(negative_mismatch.data().is_empty());
    }

    #[test]
    fn arange_rejects_zero_non_finite_and_non_advancing_steps() {
        assert_eq!(
            NDArray::arange(0_i32, 5, 0).unwrap_err(),
            AtlasNdError::InvalidArgument { op: "arange", reason: "step must be non-zero" }
        );

        assert_eq!(
            NDArray::arange(f64::NAN, 5.0, 1.0).unwrap_err(),
            AtlasNdError::InvalidArgument {
                op: "arange",
                reason: "start end and step must be finite",
            }
        );

        assert_eq!(
            NDArray::arange(0.0_f64, f64::INFINITY, 1.0).unwrap_err(),
            AtlasNdError::InvalidArgument {
                op: "arange",
                reason: "start end and step must be finite",
            }
        );

        assert_eq!(
            NDArray::arange(1.0_f64, 2.0, f64::MIN_POSITIVE / 2.0).unwrap_err(),
            AtlasNdError::InvalidArgument { op: "arange", reason: "step must advance values" }
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

    #[test]
    fn linspace_rejects_non_finite_generated_step_and_preserves_finite_degenerate_ranges() {
        assert_eq!(
            NDArray::linspace(-f32::MAX, f32::MAX, 2).unwrap_err(),
            AtlasNdError::InvalidArgument {
                op: "linspace",
                reason: "generated step must be finite",
            }
        );

        let degenerate = NDArray::linspace(f32::MAX, f32::MAX, 3).unwrap();
        assert_eq!(degenerate.shape(), &[3]);
        assert_eq!(degenerate.data(), &[f32::MAX, f32::MAX, f32::MAX]);
    }
}
