use core::any::Any;
use core::cmp::Ordering;

use atlas_ndarray::{NDArray, Numeric, checked_element_count};
use num_traits::Float;

use crate::core::{AtlasRandomError, AtlasRandomResult};

pub(crate) fn element_count(shape: &[usize]) -> AtlasRandomResult<usize> {
    Ok(checked_element_count(shape)?)
}

pub(crate) fn sample_ndarray<T, S, F>(shape: S, fill: F) -> AtlasRandomResult<NDArray<T>>
where
    T: Numeric,
    S: AsRef<[usize]>,
    F: FnOnce(&mut [T]) -> AtlasRandomResult<()>,
{
    let shape = shape.as_ref().to_vec();
    let len = element_count(&shape)?;
    let mut data = vec![T::zero(); len];
    fill(&mut data)?;

    Ok(NDArray::from_shape_vec(shape, data)?)
}

pub(crate) fn validate_uniform_bounds<T>(low: T, high: T) -> AtlasRandomResult<()>
where
    T: PartialOrd + Copy + 'static,
{
    validate_uniform_finite_bounds(low, high)?;

    if low.partial_cmp(&high) != Some(Ordering::Less) {
        return Err(AtlasRandomError::InvalidArgument {
            op: "uniform",
            reason: "low must be strictly less than high",
        });
    }

    Ok(())
}

pub(crate) fn validate_normal_parameters<T>(mean: T, stddev: T) -> AtlasRandomResult<()>
where
    T: Float,
{
    validate_normal_finite_parameters(mean, stddev)?;

    if stddev <= T::zero() {
        return Err(AtlasRandomError::InvalidArgument {
            op: "normal",
            reason: "stddev must be strictly positive",
        });
    }

    Ok(())
}

pub(crate) fn validate_bernoulli_probability(probability: f64) -> AtlasRandomResult<()> {
    if probability.is_finite() && (0.0..=1.0).contains(&probability) {
        Ok(())
    } else {
        Err(AtlasRandomError::InvalidArgument {
            op: "bernoulli",
            reason: "probability must be finite and within [0, 1]",
        })
    }
}

fn validate_uniform_finite_bounds<T>(low: T, high: T) -> AtlasRandomResult<()>
where
    T: Copy + 'static,
{
    let low_any = &low as &dyn Any;
    let high_any = &high as &dyn Any;

    if let (Some(&low), Some(&high)) =
        (low_any.downcast_ref::<f32>(), high_any.downcast_ref::<f32>())
    {
        return validate_uniform_float_bounds(low, high);
    }

    if let (Some(&low), Some(&high)) =
        (low_any.downcast_ref::<f64>(), high_any.downcast_ref::<f64>())
    {
        return validate_uniform_float_bounds(low, high);
    }

    Ok(())
}

fn validate_uniform_float_bounds<T: Float>(low: T, high: T) -> AtlasRandomResult<()> {
    if !low.is_finite() || !high.is_finite() {
        return Err(AtlasRandomError::InvalidArgument {
            op: "uniform",
            reason: "low and high must be finite",
        });
    }

    Ok(())
}

fn validate_normal_finite_parameters<T: Float>(mean: T, stddev: T) -> AtlasRandomResult<()> {
    if !mean.is_finite() || !stddev.is_finite() {
        return Err(AtlasRandomError::InvalidArgument {
            op: "normal",
            reason: "mean and stddev must be finite",
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::AtlasNdError;

    use super::{
        element_count, sample_ndarray, validate_normal_parameters, validate_uniform_bounds,
    };
    use crate::core::AtlasRandomError;

    #[test]
    fn element_count_handles_scalar_and_empty_shapes() {
        assert_eq!(element_count(&[]).unwrap(), 1);
        assert_eq!(element_count(&[0]).unwrap(), 0);
        assert_eq!(element_count(&[2, 3]).unwrap(), 6);
    }

    #[test]
    fn sample_ndarray_centralizes_shape_allocation_and_construction() {
        let array = sample_ndarray([2, 2], |data: &mut [i32]| {
            data.copy_from_slice(&[1, 2, 3, 4]);
            Ok(())
        })
        .unwrap();
        let scalar = sample_ndarray([], |data: &mut [i32]| {
            data[0] = 7;
            Ok(())
        })
        .unwrap();
        let empty = sample_ndarray([0, 3], |_data: &mut [i32]| Ok(())).unwrap();

        assert_eq!(array.shape(), &[2, 2]);
        assert_eq!(array.data(), &[1, 2, 3, 4]);
        assert_eq!(scalar.shape(), &[] as &[usize]);
        assert_eq!(scalar.data(), &[7]);
        assert_eq!(empty.shape(), &[0, 3]);
        assert!(empty.data().is_empty());
    }

    #[test]
    fn sample_ndarray_reports_shape_overflow_explicitly() {
        assert_eq!(
            sample_ndarray([usize::MAX, 2], |_data: &mut [i32]| Ok(())).unwrap_err(),
            AtlasRandomError::NdArray(AtlasNdError::ShapeOverflow {
                op: "element count",
                shape: vec![usize::MAX, 2],
            })
        );
    }

    #[test]
    fn element_count_reports_overflow_explicitly() {
        assert_eq!(
            element_count(&[usize::MAX, 2]).unwrap_err(),
            AtlasRandomError::NdArray(AtlasNdError::ShapeOverflow {
                op: "element count",
                shape: vec![usize::MAX, 2],
            })
        );
    }

    #[test]
    fn uniform_validation_reports_exact_errors() {
        assert_eq!(
            validate_uniform_bounds(4_i32, 4_i32).unwrap_err(),
            AtlasRandomError::InvalidArgument {
                op: "uniform",
                reason: "low must be strictly less than high",
            }
        );
        assert_eq!(
            validate_uniform_bounds(f64::NAN, 1.0).unwrap_err(),
            AtlasRandomError::InvalidArgument {
                op: "uniform",
                reason: "low and high must be finite",
            }
        );
        assert_eq!(
            validate_uniform_bounds(0.0_f64, f64::INFINITY).unwrap_err(),
            AtlasRandomError::InvalidArgument {
                op: "uniform",
                reason: "low and high must be finite",
            }
        );
    }

    #[test]
    fn normal_validation_reports_exact_errors() {
        assert_eq!(
            validate_normal_parameters(0.0_f64, 0.0).unwrap_err(),
            AtlasRandomError::InvalidArgument {
                op: "normal",
                reason: "stddev must be strictly positive",
            }
        );
        assert_eq!(
            validate_normal_parameters(f64::NAN, 1.0).unwrap_err(),
            AtlasRandomError::InvalidArgument {
                op: "normal",
                reason: "mean and stddev must be finite",
            }
        );
    }
}
