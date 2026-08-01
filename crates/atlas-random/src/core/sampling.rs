use core::cmp::Ordering;

use num_traits::Float;

use crate::core::{AtlasRandomError, AtlasRandomResult};

pub(crate) fn element_count(shape: &[usize]) -> usize {
    shape.iter().product()
}

pub(crate) fn validate_uniform_bounds<T>(low: T, high: T) -> AtlasRandomResult<()>
where
    T: PartialOrd,
{
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
    if !mean.is_finite() || !stddev.is_finite() {
        return Err(AtlasRandomError::InvalidArgument {
            op: "normal",
            reason: "mean and stddev must be finite",
        });
    }

    if stddev <= T::zero() {
        return Err(AtlasRandomError::InvalidArgument {
            op: "normal",
            reason: "stddev must be strictly positive",
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::core::{
        AtlasRandomError, element_count, validate_normal_parameters, validate_uniform_bounds,
    };

    #[test]
    fn element_count_handles_scalar_and_empty_shapes() {
        assert_eq!(element_count(&[]), 1);
        assert_eq!(element_count(&[0]), 0);
        assert_eq!(element_count(&[2, 3]), 6);
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
